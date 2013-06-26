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


#include <FileHelper.h>

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

int main( int argc, char *argv[] )
{
  float progress = 0.;
  float iFile = 0.;
  float nFiles;
 

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


  std::cout << std::endl << "Examining directory: " 
	    << dcmDirectoryIn << std::endl << std::endl;


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::vector< std::string > fileNames;
  niftk::GetRecursiveFilesInDirectory( dcmDirectoryIn, fileNames );

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
  typedef signed short OutputPixelType;

  const unsigned int   InputDimension = 2;

  typedef itk::Image< InternalPixelType, InputDimension > InternalImageType; 
  typedef itk::Image< OutputPixelType, InputDimension > OutputImageType;

  typedef itk::ImageFileReader< InternalImageType > ReaderType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType;


  ReaderType::Pointer reader = ReaderType::New();
  InternalImageType::Pointer image;

  for ( iterFileNames = fileNames.begin(); 
	iterFileNames < fileNames.end(); 
	++iterFileNames, iFile += 1. )
  {
    typedef itk::GDCMImageIO           ImageIOType;
    ImageIOType::Pointer gdcmImageIO = ImageIOType::New();

    progress = iFile/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;

    // Read the image

    reader->SetImageIO( gdcmImageIO );
    reader->SetFileName( *iterFileNames );
    
    try
    {
      reader->UpdateLargestPossibleRegion();
    }

    catch (itk::ExceptionObject &ex)
    {
      std::cout << "Skipping file (not DICOM?): " << *iterFileNames << std::endl 
		<< ex << std::endl << std::endl;
      continue;
    }

    std::cout << "File: " << *iterFileNames << std::endl;

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
      continue;
    }


    // Change the tag to "FOR PRESENTATION"

    ModifyTag( dictionary, "0008|0068", "FOR PRESENTATION" );
    

    // Convert the image to a "FOR PRESENTATION" version by calculating the logarithm and inverting 

    typedef itk::LogNonZeroIntensitiesImageFilter<InternalImageType, InternalImageType> LogFilterType;
    typedef itk::InvertIntensityBetweenMaxAndMinImageFilter<InternalImageType> InvertFilterType;
    typedef itk::CastImageFilter<InternalImageType, OutputImageType> CastingFilterType;
    typedef itk::MinimumMaximumImageCalculator<InternalImageType> MinimumMaximumImageCalculatorType;
    typedef itk::RescaleIntensityImageFilter< InternalImageType, InternalImageType > RescalerType;


    LogFilterType::Pointer logfilter = LogFilterType::New();
    logfilter->SetInput(image);
    logfilter->UpdateLargestPossibleRegion();

    InvertFilterType::Pointer invfilter = InvertFilterType::New();
    invfilter->SetInput(logfilter->GetOutput());
    invfilter->UpdateLargestPossibleRegion();

    CastingFilterType::Pointer caster = CastingFilterType::New();
    
    MinimumMaximumImageCalculatorType::Pointer 
      imageRangeCalculator = MinimumMaximumImageCalculatorType::New();

    imageRangeCalculator->SetImage( image );
    imageRangeCalculator->Compute();

    RescalerType::Pointer intensityRescaler = RescalerType::New();
    intensityRescaler->SetInput(invfilter->GetOutput());  

    intensityRescaler->SetOutputMinimum( 
      static_cast< InternalPixelType >( imageRangeCalculator->GetMinimum() ) );
    intensityRescaler->SetOutputMaximum( 
      static_cast< InternalPixelType >( imageRangeCalculator->GetMaximum() ) );

    intensityRescaler->UpdateLargestPossibleRegion();

    caster->SetInput( intensityRescaler->GetOutput() );

    caster->UpdateLargestPossibleRegion();


    // Create the output image filename

    fileInputFullPath = *iterFileNames;

    fileInputRelativePath = fileInputFullPath.substr( dcmDirectoryIn.length() );
     
    fileOutputRelativePath = AddPresentationFileSuffix( fileInputRelativePath,
                                                        strAdd2Suffix );
    
    fileOutputFullPath = niftk::ConcatenatePath( dcmDirectoryOut, 
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

    if ( niftk::FileExists( fileOutputFullPath ) && ( ! flgOverwrite ) )
    {
      std::cerr << std::endl << "ERROR: File " << fileOutputFullPath << " exists"
		<< std::endl << "       and can't be overwritten. Consider option: 'overwrite'."
		<< std::endl << std::endl;
      return EXIT_FAILURE;
    }
    else
    {

      WriterType::Pointer writer = WriterType::New();

      writer->SetFileName( fileOutputFullPath );
      writer->SetInput( caster->GetOutput() );
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
  }


  return EXIT_SUCCESS;
}

