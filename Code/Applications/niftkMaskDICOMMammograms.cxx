/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

/*!
 * \file niftkMaskDICOMMammograms.cxx 
 * \page niftkMaskDICOMMammograms
 * \section niftkMaskDICOMMammogramsSummary niftkMaskDICOMMammograms
 * 
 * Search for DICOM mammograms in a directory and mask them by removing any image regions in the (assumed dark) background that do not correspond to the breast region.
 *
 */


#include <niftkFileHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>

#include <itkLogHelper.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkGDCMImageIO.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>

#include <itkMammogramMaskSegmentationImageFilter.h>
#include <itkMammogramPectoralisSegmentationImageFilter.h>
#include <itkMammogramMLOorCCViewCalculator.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>

#include <vector>

#include <niftkMaskDICOMMammogramsCLP.h>


namespace fs = boost::filesystem;

typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;


struct arguments
{
  std::string dcmDirectoryIn;
  std::string dcmDirectoryOut;
  std::string strAdd2Suffix;  

  bool flgPectoralis;
  bool flgOverwrite;
  bool flgRescaleIntensitiesToMaxRange;
  bool flgVerbose;

  std::string iterFilename;
};


// -------------------------------------------------------------------------
// PrintDictionary()
// -------------------------------------------------------------------------

void PrintDictionary( DictionaryType &dictionary )
{
  DictionaryType::ConstIterator tagItr = dictionary.Begin();
  DictionaryType::ConstIterator end = dictionary.End();
   
  while ( tagItr != end )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string tagkey = tagItr->first;
      std::string tagID;
      bool found =  itk::GDCMImageIO::GetLabelFromTag( tagkey, tagID );

      std::string tagValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << tagkey << " " << tagID <<  ": " << tagValue << std::endl;
    }

    ++tagItr;
  }
};


// -------------------------------------------------------------------------
// SetTag()
// -------------------------------------------------------------------------

void SetTag( DictionaryType &dictionary,
	     std::string tagID,
	     std::string newTagValue )
{
  // Search for the tag
  
  DictionaryType::ConstIterator tagItr = dictionary.Find( tagID );
  DictionaryType::ConstIterator end = dictionary.End();
   
  if ( tagItr != end )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string tagValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << "Changing tag (" << tagID <<  ") "
		<< " from: " << tagValue 
		<< " to: " << newTagValue << std::endl;
      
      itk::EncapsulateMetaData<std::string>( dictionary, tagID, newTagValue );
    }
  }
  else
  {
    std::cout << "Setting tag (" << tagID <<  ") "
	      << " to: " << newTagValue << std::endl;
      
    itk::EncapsulateMetaData<std::string>( dictionary, tagID, newTagValue );
  }

};


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

template <class InputPixelType>
int DoMain(arguments args, InputPixelType min, InputPixelType max)
{
  bool flgInvert = false;

  enum BreastSideType { 
    UNKNOWN_BREAST_SIDE,
    LEFT_BREAST_SIDE,
    RIGHT_BREAST_SIDE,
  };

  BreastSideType breastSide = UNKNOWN_BREAST_SIDE;


  // Iterate through each file and mask it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string fileInputFullPath;
  std::string fileInputRelativePath;
  std::string fileOutputRelativePath;
  std::string fileOutputFullPath;
  std::string dirOutputFullPath;
    
  const unsigned int   InputDimension = 2;

  typedef itk::Image< InputPixelType, InputDimension > InputImageType;
  typedef itk::ImageFileReader< InputImageType > ReaderType;
  typedef itk::ImageFileWriter< InputImageType > WriterType;
  typedef itk::ImageRegionIterator< InputImageType > IteratorType;  


  typename ReaderType::Pointer reader = ReaderType::New();
  typename InputImageType::Pointer image;

  typedef itk::GDCMImageIO           ImageIOType;
  typename ImageIOType::Pointer gdcmImageIO = ImageIOType::New();

  // Read the image

  reader->SetImageIO( gdcmImageIO );
  reader->SetFileName( args.iterFilename );
    
  try
  {
    reader->UpdateLargestPossibleRegion();
  }

  catch (itk::ExceptionObject &ex)
  {
    std::cout << "Skipping file (not DICOM?): " << args.iterFilename << std::endl;
    return EXIT_FAILURE;
  }


  DictionaryType dictionary = reader->GetOutput()->GetMetaDataDictionary();
  
  if ( args.flgVerbose )
  {
    PrintDictionary( dictionary );
  }

  // Rescale the image intensities

  if ( args.flgRescaleIntensitiesToMaxRange )
  {
    itksys_ios::ostringstream value;

    typedef typename itk::RescaleIntensityImageFilter<InputImageType, 
						      InputImageType> RescaleFilterType;
    
    typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();

    rescaleFilter->SetInput( reader->GetOutput() );

    rescaleFilter->SetOutputMinimum( min );
    rescaleFilter->SetOutputMaximum( max );  
  
    std::cout << "Scaling image intensity range from: " 
	      << min << " to " << max << std::endl;

    rescaleFilter->Update();

    image = rescaleFilter->GetOutput();
    image->DisconnectPipeline();

    // Set the pixel intensity relationship sign to linear
    value.str("");
    value << "LIN";
    itk::EncapsulateMetaData<std::string>(dictionary,"0028|1040", value.str());

    // Set the pixel intensity relationship sign to one
    value.str("");
    value << 1;
    itk::EncapsulateMetaData<std::string>(dictionary,"0028|1041", value.str());

    // Set the new window centre tag value
    value.str("");
    value << max / 2;
    itk::EncapsulateMetaData<std::string>(dictionary,"0028|1050", value.str());

    // Set the new window width tag value
    value.str("");
    value << max;
    itk::EncapsulateMetaData<std::string>(dictionary,"0028|1051", value.str());

    // Set the rescale intercept and slope to zero and one 
    value.str("");
    value << 0;
    itk::EncapsulateMetaData<std::string>(dictionary, "0028|1052", value.str());
    value.str("");
    value << 1;
    itk::EncapsulateMetaData<std::string>(dictionary, "0028|1053", value.str());
  }
  else {
    image = reader->GetOutput();
    image->DisconnectPipeline();
  }


  // Check that the modality DICOM tag is 'MG'

  std::string tagModalityID = "0008|0060";
  std::string tagModalityValue;

  DictionaryType::ConstIterator tagItr = dictionary.Find( tagModalityID );
  DictionaryType::ConstIterator end = dictionary.End();
   
  if( tagItr != end )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

    if ( entryvalue )
    {
      tagModalityValue = entryvalue->GetMetaDataObjectValue();
      std::cout << "Modality Name (" << tagModalityID <<  ") "
		<< " is: " << tagModalityValue << std::endl;
    }
  }

  if ( ( tagModalityValue == std::string( "CR" ) ) || // Computed Radiography
       ( tagModalityValue == std::string( "MG" ) ) )  // Mammography
  {
    std::cout << "Image is definitely mammography - masking"
	      << std::endl;
  }
  else if ( tagModalityValue == std::string( "RG" ) )  // Radiography?
  {
    std::cout << "Image could be mammography - masking"
	      << std::endl;
  }
  else if ( ( tagModalityValue == std::string( "CT" ) ) || //  Computed Tomography
	    ( tagModalityValue == std::string( "DX" ) ) || //  Digital Radiography
	    ( tagModalityValue == std::string( "ECG" ) ) || //  Electrocardiography
	    ( tagModalityValue == std::string( "EPS" ) ) || // Cardiac Electrophysiology
	    ( tagModalityValue == std::string( "ES" ) ) || //  Endoscopy
	    ( tagModalityValue == std::string( "GM" ) ) || //  General Microscopy
	    ( tagModalityValue == std::string( "HD" ) ) || //  Hemodynamic Waveform
	    ( tagModalityValue == std::string( "IO" ) ) || //  Intra-oral Radiography
	    ( tagModalityValue == std::string( "IVUS" ) ) || //  Intravascular Ultrasound
	    ( tagModalityValue == std::string( "MR" ) ) || //  Magnetic Resonance
	    ( tagModalityValue == std::string( "NM" ) ) || //  Nuclear Medicine
	    ( tagModalityValue == std::string( "OP" ) ) || //  Ophthalmic Photography
	    ( tagModalityValue == std::string( "PT" ) ) || //  Positron emission tomography
	    ( tagModalityValue == std::string( "PX" ) ) || //  Panoramic X-Ray
	    ( tagModalityValue == std::string( "RF" ) ) || //  Radiofluoroscopy
	    ( tagModalityValue == std::string( "RTIMAGE" ) ) || //  Radiotherapy Image
	    ( tagModalityValue == std::string( "SM" ) ) || //  Slide Microscopy
	    ( tagModalityValue == std::string( "US" ) ) || //  Ultrasound
	    ( tagModalityValue == std::string( "XA" ) ) || //  X-Ray Angiography
	    ( tagModalityValue == std::string( "XC" ) ) ) //  External-camera Photography
  {
    std::cout << "Skipping image - does not appear to be a mammogram" << std::endl << std::endl;
    return EXIT_SUCCESS;
  }
  else
  {
    std::cout << "WARNING: Unsure if this ia a mammogram but masking anyway" 
	      << std::endl;
  }


  // Check if the DICOM Inverse tag is set

  std::string tagInverse = "2050|0020";
  
  DictionaryType::ConstIterator tagInverseItr = dictionary.Find( tagInverse );
  DictionaryType::ConstIterator tagInverseEnd = dictionary.End();
  
  if ( tagInverseItr != tagInverseEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagInverseItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string strInverse( "INVERSE" );
      std::string tagInverseValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << "Tag (" << tagInverse 
		<< ") is: " << tagInverseValue << std::endl;

      std::size_t foundInverse = tagInverseValue.find( strInverse );
      if (foundInverse != std::string::npos)
      {
	flgInvert = true;
	std::cout << "Image is INVERSE - inverting" << std::endl;
	SetTag( dictionary, tagInverse, "IDENTITY" );
      }
    }
  }


  // Fix the MONOCHROME1 issue

  std::string tagPhotoInterpID = "0028|0004";
  
  DictionaryType::ConstIterator tagPhotoInterpItr = dictionary.Find( tagPhotoInterpID );
  DictionaryType::ConstIterator tagPhotoInterpEnd = dictionary.End();
  
  if ( tagPhotoInterpItr != tagPhotoInterpEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagPhotoInterpItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string strMonochrome1( "MONOCHROME1" );
      std::string tagPhotoInterpValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << "Tag (" << tagPhotoInterpID 
		<< ") is: " << tagPhotoInterpValue << std::endl;

      std::size_t foundMonochrome1 = tagPhotoInterpValue.find( strMonochrome1 );
      if (foundMonochrome1 != std::string::npos)
      {
	flgInvert = true;
	std::cout << "Image is MONOCHROME1 - inverting" << std::endl;
	SetTag( dictionary, tagPhotoInterpID, "MONOCHROME2" );
      }
    }
  }

  // Invert the image

  if ( flgInvert )
  {
    typedef typename itk::InvertIntensityBetweenMaxAndMinImageFilter<InputImageType> InvertFilterType;
    
    typename InvertFilterType::Pointer invertFilter = InvertFilterType::New();
    invertFilter->SetInput( image );

    invertFilter->Update( );
	
    image = invertFilter->GetOutput();
    image->DisconnectPipeline();
  }


  // Create the segmentation filter

  typedef unsigned char MaskPixelType;
  typedef typename itk::Image< MaskPixelType, InputDimension > MaskImageType;   

  typedef typename itk::MammogramMaskSegmentationImageFilter<InputImageType, MaskImageType> MammogramMaskSegmentationImageFilterType;

  typename MammogramMaskSegmentationImageFilterType::Pointer segFilter = MammogramMaskSegmentationImageFilterType::New();

  segFilter->SetInput( image );

  segFilter->SetVerbose( args.flgVerbose );

  try {
    segFilter->Update();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to segment image" << std::endl
              << err << std::endl; 
    return EXIT_FAILURE;
  }                

  typename MaskImageType::Pointer mask = segFilter->GetOutput();

  mask->DisconnectPipeline();


  // Detect the pectoral muscle also?

  if ( args.flgPectoralis )
  {
    typedef typename itk::MammogramMLOorCCViewCalculator< InputImageType > 
      MammogramMLOorCCViewCalculatorType;
    
    typedef typename MammogramMLOorCCViewCalculatorType::MammogramViewType MammogramViewType;

    MammogramViewType mammogramView = MammogramMLOorCCViewCalculatorType::UNKNOWN_MAMMO_VIEW;
    double mammogramViewScore = 0;

    typename MammogramMLOorCCViewCalculatorType::Pointer 
      viewCalculator = MammogramMLOorCCViewCalculatorType::New();

    viewCalculator->SetImage( image );
    viewCalculator->SetDictionary( dictionary );
    viewCalculator->SetImageFileName( args.iterFilename );

    viewCalculator->SetVerbose( args.flgVerbose );

    try {
      viewCalculator->Compute();
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to compute left or right breast" << std::endl
                << err << std::endl; 
      return EXIT_FAILURE;
    }                

    mammogramView = viewCalculator->GetMammogramView();

    if ( mammogramView == MammogramMLOorCCViewCalculatorType::MLO_MAMMO_VIEW )
    {
      typedef itk::MammogramPectoralisSegmentationImageFilter<InputImageType, MaskImageType> 
        MammogramPectoralisSegmentationImageFilterType;

      typename MammogramPectoralisSegmentationImageFilterType::Pointer 
        pecFilter = MammogramPectoralisSegmentationImageFilterType::New();

      pecFilter->SetInput( image );  
      
      pecFilter->SetVerbose( args.flgVerbose );

      if ( mask )
      {
        pecFilter->SetMask( mask );
      }
      
      try
      {
        pecFilter->Update(); 
      }
      catch( itk::ExceptionObject & err ) 
      { 
        std::cerr << "Failed to segment the pectoral muscle: " << err << std::endl; 
        return EXIT_FAILURE;
      }                

      typename MaskImageType::Pointer pecMask = pecFilter->GetOutput();

      pecMask->DisconnectPipeline();

      typename itk::ImageRegionIterator< MaskImageType > 
        maskIterator( mask, mask->GetLargestPossibleRegion());

      typename itk::ImageRegionConstIterator< MaskImageType > 
        pecIterator( pecMask, pecMask->GetLargestPossibleRegion());
       
      for ( maskIterator.GoToBegin(), pecIterator.GoToBegin();
            ! maskIterator.IsAtEnd();
            ++maskIterator, ++pecIterator )
      {
        if ( pecIterator.Get() )
          maskIterator.Set( 0 );
      }
    }
  }


  // Apply the mask to the image

  typename itk::ImageRegionConstIterator< MaskImageType > 
    inputIterator( mask, mask->GetLargestPossibleRegion());

  typename itk::ImageRegionIterator< InputImageType > 
    outputIterator(image, image->GetLargestPossibleRegion());
        
  for ( inputIterator.GoToBegin(), outputIterator.GoToBegin();
        ! inputIterator.IsAtEnd();
        ++inputIterator, ++outputIterator )
  {
    if ( ! inputIterator.Get() )
      outputIterator.Set( 0 );
  }


  // Create the output image filename

  fileInputFullPath = args.iterFilename;

  fileInputRelativePath = fileInputFullPath.substr( args.dcmDirectoryIn.length() );
     
  fileOutputRelativePath = niftk::AddStringToImageFileSuffix( fileInputRelativePath,
                                                              args.strAdd2Suffix );
    
  fileOutputFullPath = niftk::ConcatenatePath( args.dcmDirectoryOut, 
					       fileOutputRelativePath );

  dirOutputFullPath = fs::path( fileOutputFullPath ).branch_path().string();
    
  if ( ! niftk::DirectoryExists( dirOutputFullPath ) )
  {
    niftk::CreateDirAndParents( dirOutputFullPath );
  }
      
  std::cout << "Input relative filename: " << fileInputRelativePath << std::endl
	    << "Output relative filename: " << fileOutputRelativePath << std::endl
	    << "Output directory: " << dirOutputFullPath << std::endl;


  // Write the image to the output file

  if ( niftk::FileExists( fileOutputFullPath ) && ( ! args.flgOverwrite ) )
  {
    std::cerr << std::endl << "ERROR: File " << fileOutputFullPath << " exists"
	      << std::endl << "       and can't be overwritten. Consider option: 'overwrite'."
	      << std::endl << std::endl;
    return EXIT_FAILURE;
  }
  else
  {
    if ( args.flgVerbose )
    {
      PrintDictionary( dictionary );
    }

    typename WriterType::Pointer writer = WriterType::New();

    writer->SetFileName( fileOutputFullPath );

    image->DisconnectPipeline();
    writer->SetInput( image );

    gdcmImageIO->SetMetaDataDictionary( dictionary );
    gdcmImageIO->KeepOriginalUIDOn( );
    writer->SetImageIO( gdcmImageIO );

    writer->UseInputMetaDataDictionaryOff();

    try
    {
      std::cout << "Writing image to file: " << fileOutputFullPath << std::endl;
      writer->Update();
    }
    catch (itk::ExceptionObject & e)
    {
      std::cerr << "ERROR: Failed to write image: " << std::endl << e << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << std::endl;

  return EXIT_SUCCESS;
}


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  float progress = 0.;
  float iFile = 0.;
  float nFiles;

  struct arguments args;

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( dcmDirectoryIn.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    std::cerr << "ERROR: The input directory must be specified" << std::endl;
    return EXIT_FAILURE;
  }

  if ( dcmDirectoryOut.length() == 0 )
  {
    dcmDirectoryOut = dcmDirectoryIn;
  }

  args.dcmDirectoryIn  = dcmDirectoryIn;                     
  args.dcmDirectoryOut = dcmDirectoryOut;                    

  args.strAdd2Suffix = strAdd2Suffix;                      
				   	                                                 
  args.flgOverwrite  = flgOverwrite;                       
  args.flgVerbose    = flgVerbose;    
  args.flgPectoralis = flgPectoralis;

  args.flgRescaleIntensitiesToMaxRange = flgRescaleIntensitiesToMaxRange;


  std::cout << std::endl << "Examining directory: " 
	    << dcmDirectoryIn << std::endl << std::endl;


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::vector< std::string > fileNames;
  std::vector< std::string >::iterator iterFileNames;       

  niftk::GetRecursiveFilesInDirectory( dcmDirectoryIn, fileNames );

  nFiles = fileNames.size();

  for ( iterFileNames = fileNames.begin(); 
	iterFileNames < fileNames.end(); 
	++iterFileNames, iFile += 1. )
  {
    args.iterFilename = *iterFileNames;
    
    std::cout << "File: " << args.iterFilename << std::endl;

    progress = iFile/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;

  
    itk::ImageIOBase::Pointer imageIO;
    imageIO = itk::ImageIOFactory::CreateImageIO(args.iterFilename.c_str(), 
						 itk::ImageIOFactory::ReadMode);

    if ( ( ! imageIO ) || ( ! imageIO->CanReadFile( args.iterFilename.c_str() ) ) )
    {
      std::cerr << "WARNING: Unrecognised image type, skipping file: " 
		<< args.iterFilename << std::endl;
      continue;
    }


    int result;

    switch (itk::PeekAtComponentType(args.iterFilename))
    {
    case itk::ImageIOBase::UCHAR:
      result = DoMain<unsigned char>( args,
                                      itk::NumericTraits<unsigned char>::ZeroValue(),
                                      itk::NumericTraits<unsigned char>::max() );  
      break;
    
    case itk::ImageIOBase::CHAR:
      result = DoMain<char>( args,
                             itk::NumericTraits<char>::ZeroValue(),
                             itk::NumericTraits<char>::max() );  
      break;

    case itk::ImageIOBase::USHORT:
      result = DoMain<unsigned short>( args,
                                       itk::NumericTraits<unsigned short>::ZeroValue(),
                                       static_cast<unsigned short>( 32767 ) );
      break;

    case itk::ImageIOBase::SHORT:
      result = DoMain<short>( args,
                              itk::NumericTraits<short>::ZeroValue(),
                              static_cast<short>( 32767 ) );  
      break;

    case itk::ImageIOBase::UINT:
      result = DoMain<unsigned int>( args,
                                     itk::NumericTraits<unsigned int>::ZeroValue(),
                                     static_cast<unsigned int>( 32767 ) );  
      break;

    case itk::ImageIOBase::INT:
      result = DoMain<int>( args,
                            itk::NumericTraits<int>::ZeroValue(),
                            static_cast<int>( 32767 ) );  
      break;

    case itk::ImageIOBase::ULONG:
      result = DoMain<unsigned long>( args,
                                      itk::NumericTraits<unsigned long>::ZeroValue(),
                                      static_cast<unsigned long>( 32767 ) );  
      break;

    case itk::ImageIOBase::LONG:
      result = DoMain<long>( args,
                             itk::NumericTraits<long>::ZeroValue(),
                             static_cast<long>( 32767 ) );  
      break;

    case itk::ImageIOBase::FLOAT:
      result = DoMain<float>( args,
                              itk::NumericTraits<float>::ZeroValue(),
                              static_cast<float>( 32767 ) );  
      break;

    case itk::ImageIOBase::DOUBLE:
      result = DoMain<double>( args,
                               itk::NumericTraits<double>::ZeroValue(),
                               static_cast<double>( 32767 ) );  
      break;

    default:
      std::cerr << "WARNING: Unrecognised pixel type, skipping file: " 
		<< args.iterFilename << std::endl;
    }

    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}
 
 

