/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/*!
 * \file niftkAnonymiseDICOMMammograms.cxx 
 * \page niftkAnonymiseDICOMMammograms
 * \section niftkAnonymiseDICOMMammogramsSummary niftkAnonymiseDICOMMammograms
 * 
 * Search for DICOM mammograms in a directory and anonymise them by removing patient information from the DICOM header and/or applying a rectangular mask to remove the label.
 *
 */


#include "FileHelper.h"

#include "itkLogHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"
#include "itkGDCMImageIO.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"

#include <vector>

#include "niftkAnonymiseDICOMMammogramsCLP.h"


namespace fs = boost::filesystem;

typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;



// -------------------------------------------------------------------------
// AddAnonymousFileSuffix
// -------------------------------------------------------------------------

std::string AddAnonymousFileSuffix( std::string fileName, std::string strAdd2Suffix )
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
// AnonymiseTag()
// -------------------------------------------------------------------------

void AnonymiseTag( bool flgDontAnonymise, 
		    DictionaryType &dictionary,
		    std::string tagID,
		    std::string newTagValue )
{
  if ( flgDontAnonymise )
    return;

  // Search for the tag
  
  std::string tagModalityValue;
  
  DictionaryType::ConstIterator tagItr = dictionary.Find( tagID );
  DictionaryType::ConstIterator end = dictionary.End();
   
  if ( tagItr != end )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string tagModalityValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << "Anonymising tag (" << tagID <<  ") "
		<< " from: " << tagModalityValue 
		<< " to: " << newTagValue << std::endl;
      
      itk::EncapsulateMetaData<std::string>( dictionary, tagID, newTagValue );
    }
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

  if ( ! ( flgAnonymiseDICOMHeader || flgAnonymiseImageLabel ) )
  {
    commandLine.getOutput()->usage(commandLine);
    std::cerr << "ERROR: Please specify at least one of 'anonHeader' or 'anonLabel'" << std::endl;
    return EXIT_FAILURE;
  }

  if ( dcmDirectoryOut.length() == 0 )
  {
    dcmDirectoryOut = dcmDirectoryIn;
  }

  dcmDirectoryIn  = dcmDirectoryIn;
  dcmDirectoryOut = dcmDirectoryOut;

  strAdd2Suffix = strAdd2Suffix;

  flgOverwrite = flgOverwrite;

  flgAnonymiseDICOMHeader = flgAnonymiseDICOMHeader;
  flgAnonymiseImageLabel  = flgAnonymiseImageLabel;

  labelWidth  = labelWidth;
  labelHeight = labelHeight;

  labelPosition  = labelPosition;


  std::cout << std::endl << "Examining directory: " 
	    << dcmDirectoryIn << std::endl << std::endl;


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::vector< std::string > fileNames;
  niftk::GetRecursiveFilesInDirectory( dcmDirectoryIn, fileNames );

  nFiles = fileNames.size();


  // Iterate through each file and anonymise it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string fileInputFullPath;
  std::string fileInputRelativePath;
  std::string fileOutputRelativePath;
  std::string fileOutputFullPath;
  std::string dirOutputFullPath;
    
  std::vector< std::string >::iterator iterFileNames;       

  typedef signed short InputPixelType;
  const unsigned int   InputDimension = 2;

  typedef itk::Image< InputPixelType, InputDimension > InputImageType;
  typedef itk::ImageFileReader< InputImageType > ReaderType;
  typedef itk::ImageFileWriter< InputImageType > WriterType;
  typedef itk::ImageRegionIterator< InputImageType > IteratorType;  


  ReaderType::Pointer reader = ReaderType::New();
  InputImageType::Pointer image;

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
	std::string tagModalityValue = entryvalue->GetMetaDataObjectValue();
	std::cout << "Modality Name (" << tagModalityID <<  ") "
		  << " is: " << tagModalityValue << std::endl;
      }
    }

    if ( ( tagModalityValue == std::string( "CR" ) ) || //  Computed Radiography
	 ( tagModalityValue == std::string( "MG" ) ) )  //  Mammography
    {
      std::cout << "Image is definitely mammography - anonymising"
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
	      ( tagModalityValue == std::string( "RG" ) ) || //  Radiographic imaging
	      ( tagModalityValue == std::string( "RTIMAGE" ) ) || //  Radiotherapy Image
	      ( tagModalityValue == std::string( "SM" ) ) || //  Slide Microscopy
	      ( tagModalityValue == std::string( "US" ) ) || //  Ultrasound
	      ( tagModalityValue == std::string( "XA" ) ) || //  X-Ray Angiography
	      ( tagModalityValue == std::string( "XC" ) ) ) //  External-camera Photography
    {
      std::cout << "Skipping image - does not appear to be a mammogram" << std::endl << std::endl;
      continue;
    }
    else
    {
      std::cout << "WARNING: Unsure if this ia a mammogram but anonymising anyway" 
		<< std::endl;
    }

    // Set the label region to zero

    if ( flgAnonymiseImageLabel )
    {
      InputImageType::RegionType region;
      InputImageType::SizeType size;
      InputImageType::IndexType start;

      region = image->GetLargestPossibleRegion();      

      size = region.GetSize();

      std::cout << "Image size: " << size << std::endl;

      start[0] = size[0];
      start[1] = size[1];

      size[0] = static_cast<unsigned int>( static_cast<float>(size[0]) * labelWidth/100. );
      size[1] = static_cast<unsigned int>( static_cast<float>(size[1]) * labelHeight/100. );

      if ( labelPosition == std::string( "Top-Left" ) )
      {
	start[0] = 0;
	start[1] = 0;
      }
      else if ( labelPosition == std::string( "Top-Right" ) )
      {
	start[0] -= size[0];
	start[1] = 0;
      }
      else if ( labelPosition == std::string( "Bottom-Right" ) )
      {
	start[0] -= size[0];
	start[1] -= size[1];
      }
      else if ( labelPosition == std::string( "Bottom-Left" ) )
      {
	start[0] = 0;
	start[1] -= size[1];
      }

      region.SetSize( size );
      region.SetIndex( start );

      std::cout << "Removing label from region: " << region << std::endl;

      IteratorType itLabel( image, region );
  
      for ( itLabel.GoToBegin(); 
	    ! itLabel.IsAtEnd() ; 
	    ++itLabel )
      {
	itLabel.Set( 0 );
      }
    }


    // Anonymise the DICOM header?

    if ( flgAnonymiseDICOMHeader )
    {
      AnonymiseTag( flgDontAnonPatientsName,  			              dictionary, "0010|0010", "Anonymous"    ); // Patient's Name                               
      AnonymiseTag( flgDontAnonPatientsBirthDate,			      dictionary, "0010|0030", "00-00-00"     ); // Patient's Birth Date                        
      AnonymiseTag( flgDontAnonOtherPatientNames, 			      dictionary, "0010|1001", "None"         ); // Other Patient Names                         
      AnonymiseTag( flgDontAnonPatientsBirthName, 			      dictionary, "0010|1005", "Anonymous"    ); // Patient's Birth Name                        
      AnonymiseTag( flgDontAnonPatientsAddress, 			      dictionary, "0010|1040", "None"         ); // Patient's Address                           
      AnonymiseTag( flgDontAnonPatientsMothersBirthName, 		      dictionary, "0010|1060", "Anonymous"    ); // Patient's Mother's Birth Name               
      AnonymiseTag( flgDontAnonPatientsTelephoneNumbers, 		      dictionary, "0010|2154", "None"         ); // Patient's Telephone Numbers                 
    }
      

    // Create the output image filename

    fileInputFullPath = *iterFileNames;

    fileInputRelativePath = fileInputFullPath.substr( dcmDirectoryIn.length() );
     
    fileOutputRelativePath = AddAnonymousFileSuffix( fileInputRelativePath,
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
      writer->SetInput( image );
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

