/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

/*!
 * \file niftkConvertRawDICOMMammogramsToPresentation.cxx 
 * \page niftkConvertRawDICOMMammogramsToPresentation
 * \section niftkConvertRawDICOMMammogramsToPresentationSummary niftkConvertRawDICOMMammogramsToPresentation
 * 
 * Search for raw "For Processing" DICOM mammograms in a directory and convert them to "For Presentation" versions by calculating the logarithm of their intensities and then the inverse.
 *
 */


#include <niftkFileHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>

#include <itkLogHelper.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkGDCMImageIO.h>
#include <itkLogNonZeroIntensitiesImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>

#include <vector>

#include <niftkConvertRawDICOMMammogramsToPresentationCLP.h>


namespace fs = boost::filesystem;

typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;



struct arguments
{
  std::string dcmDirectoryIn;
  std::string dcmDirectoryOut;
  std::string strAdd2Suffix;  

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
// ModifyTag()
// -------------------------------------------------------------------------

void ModifyTag( DictionaryType &dictionary,
                std::string tagID,
                std::string newTagValue )
{
  // Search for the tag
  
  std::string tagValue;
  
  DictionaryType::ConstIterator tagItr = dictionary.Find( tagID );
  DictionaryType::ConstIterator tagEnd = dictionary.End();
   
  if ( tagItr != tagEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string tagValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << "Modifying tag (" << tagID <<  ") "
		<< " from: " << tagValue 
		<< " to: " << newTagValue << std::endl;
      
      itk::EncapsulateMetaData<std::string>( dictionary, tagID, newTagValue );
    }
  }

};


// -------------------------------------------------------------------------
// AddPresentationFileSuffix
// -------------------------------------------------------------------------

std::string AddPresentationFileSuffix( std::string fileName, std::string strAdd2Suffix )
{
  std::string suffix;
  std::string newSuffix;

  if ( ( fileName.length() >= 4 ) && 
       ( fileName.substr( fileName.length() - 4 ) == std::string( ".dcm" ) ) )
  {
    suffix = std::string( ".dcm" );
  }

  else if ( ( fileName.length() >= 4 ) && 
            ( fileName.substr( fileName.length() - 4 ) == std::string( ".DCM" ) ) )
  {
    suffix = std::string( ".DCM" );
  }

  else if ( ( fileName.length() >= 6 ) && 
            ( fileName.substr( fileName.length() - 6 ) == std::string( ".dicom" ) ) )
  {
    suffix = std::string( ".dicom" );
  }

  else if ( ( fileName.length() >= 6 ) && 
            ( fileName.substr( fileName.length() - 6 ) == std::string( ".DICOM" ) ) )
  {
    suffix = std::string( ".DICOM" );
  }

  else if ( ( fileName.length() >= 4 ) && 
            ( fileName.substr( fileName.length() - 4 ) == std::string( ".IMA" ) ) )
  {
    suffix = std::string( ".IMA" );
  }

  std::cout << "Suffix: '" << suffix << "'" << std::endl;

  newSuffix = strAdd2Suffix + suffix;

  if ( ( fileName.length() >= newSuffix.length() ) && 
       ( fileName.substr( fileName.length() - newSuffix.length() ) != newSuffix ) )
  {
    return fileName.substr( 0, fileName.length() - suffix.length() ) + newSuffix;
  }
  else
  {
    return fileName;
  }
};


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

template <class OutputPixelType>
int DoMain(arguments args)
{
  bool flgPreInvert = false;

  float progress = 0.;
  float iFile = 0.;
  float nFiles;
 
  itksys_ios::ostringstream value;


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::vector< std::string > fileNames;
  niftk::GetRecursiveFilesInDirectory( args.dcmDirectoryIn, fileNames );

  nFiles = fileNames.size();


  // Iterate through each file and convert it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string fileInputFullPath;
  std::string fileInputRelativePath;
  std::string fileOutputRelativePath;
  std::string fileOutputFullPath;
  std::string dirOutputFullPath;
    
  std::vector< std::string >::iterator iterFileNames;       

  typedef float InternalPixelType;

  const unsigned int   InputDimension = 2;

  typedef itk::Image< InternalPixelType, InputDimension > InternalImageType; 
  typedef itk::Image< OutputPixelType, InputDimension > OutputImageType;

  typedef itk::LogNonZeroIntensitiesImageFilter<InternalImageType, InternalImageType> LogFilterType;
  typedef itk::InvertIntensityBetweenMaxAndMinImageFilter<InternalImageType> InvertFilterType;
  typedef itk::CastImageFilter<InternalImageType, OutputImageType> CastingFilterType;
  typedef itk::MinimumMaximumImageCalculator<InternalImageType> MinimumMaximumImageCalculatorType;
  typedef itk::RescaleIntensityImageFilter< InternalImageType, InternalImageType > RescalerType;
 
  typedef itk::ImageFileReader< InternalImageType > ReaderType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType;


  ReaderType::Pointer reader = ReaderType::New();
  InternalImageType::Pointer image;

  typedef itk::GDCMImageIO           ImageIOType;
  ImageIOType::Pointer gdcmImageIO = ImageIOType::New();

  progress = iFile/nFiles;
  std::cout << "<filter-progress>" << std::endl
            << progress << std::endl
            << "</filter-progress>" << std::endl;

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

  std::cout << "File: " << args.iterFilename << std::endl;

  image = reader->GetOutput();
  image->DisconnectPipeline();

  DictionaryType &dictionary = image->GetMetaDataDictionary();
  
  DictionaryType::ConstIterator tagItr;
  DictionaryType::ConstIterator tagEnd;

  // Check that the modality DICOM tag is 'MG'

  std::string tagModalityID = "0008|0060";
  std::string tagModalityValue;

  tagItr = dictionary.Find( tagModalityID );
  tagEnd = dictionary.End();
   
  if( tagItr != tagEnd )
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

  // Check that the 'Presentation Intent Type' is 'For Processing'

  std::string tagForProcessingID = "0008|0068";
  std::string tagForProcessingValue;

  tagItr = dictionary.Find( tagForProcessingID );
  tagEnd = dictionary.End();
   
  if( tagItr != tagEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

    if ( entryvalue )
    {
      tagForProcessingValue = entryvalue->GetMetaDataObjectValue();
      std::cout << "Presentation Intent Type (" << tagForProcessingID <<  ") "
                << " is: " << tagForProcessingValue << std::endl;
    }
  }

  // Process this file?

  if ( ( ( tagModalityValue == std::string( "CR" ) ) ||    //  Computed Radiography
         ( tagModalityValue == std::string( "MG" ) ) ) &&  //  Mammography
       ( tagForProcessingValue == std::string( "FOR PROCESSING" ) ) )
  {
    std::cout << "Image is a raw \"FOR PROCESSING\" mammogram - converting"
              << std::endl;
  }
  else
  {
    std::cout << "Skipping image - does not appear to be a \"FOR PROCESSING\" mammogram" 
              << std::endl << std::endl;
    return EXIT_FAILURE;
  }


  // Change the tag to "FOR PRESENTATION"

  ModifyTag( dictionary, "0008|0068", "FOR PRESENTATION" );

  // Set the pixel intensity relationship sign to linear
  value.str("");
  value << "LIN";
  itk::EncapsulateMetaData<std::string>(dictionary,"0028|1040", value.str());

  // Set the pixel intensity relationship sign to one
  value.str("");
  value << 1;
  itk::EncapsulateMetaData<std::string>(dictionary,"0028|1041", value.str());

  // Set the presentation LUT shape
  ModifyTag( dictionary, "2050|0020", "IDENITY" );
    
  // Check whether this is MONOCHROME1 or 2 and hence whether to invert

  std::string tagPhotoInterpID = "0028|0004";
  std::string tagPhotoInterpValue;

  tagItr = dictionary.Find( tagPhotoInterpID );
  tagEnd = dictionary.End();
   
  if( tagItr != tagEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

    if ( entryvalue )
    {
      tagPhotoInterpValue = entryvalue->GetMetaDataObjectValue();
      std::cout << "Photometric interportation is (" << tagPhotoInterpID <<  ") "
                << " is: " << tagPhotoInterpValue << std::endl;
    }
  }

  std::size_t found = tagPhotoInterpValue.find( "MONOCHROME2" );
  if ( found != std::string::npos )
  {
    std::cout << "Image is \"MONOCHROME2\" so will not be inverted"
              << std::endl;
    flgPreInvert = true;        // Actually we pre-invert it
  }

  found = tagPhotoInterpValue.find( "MONOCHROME1" );
  if ( found != std::string::npos )
  {
    ModifyTag( dictionary, "0028|0004", "MONOCHROME2" );
  }

   
  // Set the desired output range (i.e. the same as the input)

  MinimumMaximumImageCalculatorType::Pointer 
    imageRangeCalculator = MinimumMaximumImageCalculatorType::New();

  imageRangeCalculator->SetImage( image );
  imageRangeCalculator->Compute();

  RescalerType::Pointer intensityRescaler = RescalerType::New();

  if ( args.flgRescaleIntensitiesToMaxRange )
  {
    intensityRescaler->SetOutputMinimum( itk::NumericTraits<OutputPixelType>::ZeroValue() );
    intensityRescaler->SetOutputMaximum( itk::NumericTraits<OutputPixelType>::max() );

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
    value << itk::NumericTraits<OutputPixelType>::max() / 2;
    itk::EncapsulateMetaData<std::string>(dictionary,"0028|1050", value.str());

    // Set the new window width tag value
    value.str("");
    value << itk::NumericTraits<OutputPixelType>::max();
    itk::EncapsulateMetaData<std::string>(dictionary,"0028|1051", value.str());

    // Set the rescale intercept and slope to zero and one 
    value.str("");
    value << 0;
    itk::EncapsulateMetaData<std::string>(dictionary, "0028|1052", value.str());
    value.str("");
    value << 1;
    itk::EncapsulateMetaData<std::string>(dictionary, "0028|1053", value.str());
  }
  else
  {
    intensityRescaler->SetOutputMinimum( 
      static_cast< InternalPixelType >( imageRangeCalculator->GetMinimum() ) );
    intensityRescaler->SetOutputMaximum( 
      static_cast< InternalPixelType >( imageRangeCalculator->GetMaximum() ) );
  }


  std::cout << "Image output range will be: " << intensityRescaler->GetOutputMinimum()
            << " to " << intensityRescaler->GetOutputMaximum() << std::endl;


  // Convert the image to a "FOR PRESENTATION" version by calculating the logarithm and inverting 

  if ( flgPreInvert ) 
  {
    InvertFilterType::Pointer invfilter = InvertFilterType::New();
    invfilter->SetInput( image );
    invfilter->UpdateLargestPossibleRegion();
    image = invfilter->GetOutput();
  }

  LogFilterType::Pointer logfilter = LogFilterType::New();
  logfilter->SetInput(image);
  logfilter->UpdateLargestPossibleRegion();
   
  InvertFilterType::Pointer invfilter = InvertFilterType::New();
  invfilter->SetInput(logfilter->GetOutput());
  invfilter->UpdateLargestPossibleRegion();
  image = invfilter->GetOutput();
      

  typename CastingFilterType::Pointer caster = CastingFilterType::New();

  intensityRescaler->SetInput( image );  

  intensityRescaler->UpdateLargestPossibleRegion();

  caster->SetInput( intensityRescaler->GetOutput() );

  caster->UpdateLargestPossibleRegion();


  // Create the output image filename

  fileInputFullPath = args.iterFilename;

  fileInputRelativePath = fileInputFullPath.substr( args.dcmDirectoryIn.length() );
     
  fileOutputRelativePath = AddPresentationFileSuffix( fileInputRelativePath,
                                                      args.strAdd2Suffix );
    
  fileOutputFullPath = niftk::ConcatenatePath( args.dcmDirectoryOut, 
                                               fileOutputRelativePath );

  dirOutputFullPath = fs::path( fileOutputFullPath ).branch_path().string();
    
  if ( ! niftk::DirectoryExists( dirOutputFullPath ) )
  {
    niftk::CreateDirectoryAndParents( dirOutputFullPath );
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

    typename OutputImageType::Pointer outImage = caster->GetOutput();
    outImage->DisconnectPipeline();
    outImage->SetMetaDataDictionary( dictionary );

    writer->SetFileName( fileOutputFullPath );
    writer->SetInput( outImage );
    writer->SetImageIO( gdcmImageIO );

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
				   	                                                 
  args.flgOverwrite            = flgOverwrite;                       
  args.flgVerbose              = flgVerbose;    

  args.flgRescaleIntensitiesToMaxRange = flgRescaleIntensitiesToMaxRange;


  std::cout << std::endl << "Examining directory: " 
	    << args.dcmDirectoryIn << std::endl << std::endl;


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
      result = DoMain<unsigned char>(args);  
      break;
    
    case itk::ImageIOBase::CHAR:
      result = DoMain<char>(args);  
      break;

    case itk::ImageIOBase::USHORT:
      result = DoMain<unsigned short>(args);  
      break;

    case itk::ImageIOBase::SHORT:
      result = DoMain<short>(args);  
      break;

    case itk::ImageIOBase::UINT:
      result = DoMain<unsigned int>(args);  
      break;

    case itk::ImageIOBase::INT:
      result = DoMain<int>(args);  
      break;

    case itk::ImageIOBase::ULONG:
      result = DoMain<unsigned long>(args);  
      break;

    case itk::ImageIOBase::LONG:
      result = DoMain<long>(args);  
      break;

    case itk::ImageIOBase::FLOAT:
      result = DoMain<float>(args);  
      break;

    case itk::ImageIOBase::DOUBLE:
      result = DoMain<double>(args);  
      break;

    default:
      std::cerr << "WARNING: Unrecognised pixel type, skipping file: " 
		<< args.iterFilename << std::endl;
    }

    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}
 
 

