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
 * \section niftkAnonymiseDICOMMammograms Search for DICOM mammograms in a
 * directory and anonymise them by removing patient information from
 * the DICOM header and/or applying a rectangular mask to remove the
 * label.
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

    typedef itk::MetaDataDictionary DictionaryType;
    DictionaryType &dictionary = image->GetMetaDataDictionary();


    // Check that the modality DICOM tag is 'MG'

    typedef itk::MetaDataObject< std::string > MetaDataStringType;

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

      if ( ! flgDontAnonPatientsName )                                itk::EncapsulateMetaData<std::string>( dictionary, "0010|0010", "Anonymous"    ); // Patient's Name                               
      if ( ! flgDontAnonPatientID ) 				      itk::EncapsulateMetaData<std::string>( dictionary, "0010|0020", "No Identifier"); // Patient ID                                  
      if ( ! flgDontAnonIssuerofPatientID ) 			      itk::EncapsulateMetaData<std::string>( dictionary, "0010|0021", "No Issuer"    ); // Issuer of Patient ID                        
      if ( ! flgDontAnonPatientsBirthDate ) 			      itk::EncapsulateMetaData<std::string>( dictionary, "0010|0030", "00-00-00"     ); // Patient's Birth Date                        
      if ( ! flgDontAnonPatientsBirthTime ) 			      itk::EncapsulateMetaData<std::string>( dictionary, "0010|0032", "00:00"        ); // Patient's Birth Time                        
      if ( ! flgDontAnonPatientsSex ) 				      itk::EncapsulateMetaData<std::string>( dictionary, "0010|0040", "M/F"          ); // Patient's Sex                               
      if ( ! flgDontAnonPatientsInsurancePlanCodeSequence ) 	      itk::EncapsulateMetaData<std::string>( dictionary, "0010|0050", "None"         ); // Patient's Insurance Plan Code Sequence      
      if ( ! flgDontAnonPatientsPrimaryLanguageCodeSequence ) 	      itk::EncapsulateMetaData<std::string>( dictionary, "0010|0101", "None"         ); // Patient's Primary Language Code Sequence    
      if ( ! flgDontAnonPatientsPrimaryLanguageCodeModifierSequence ) itk::EncapsulateMetaData<std::string>( dictionary, "0010|0102", "None"         ); // Patient's Primary Language Code Modifier Seq
      if ( ! flgDontAnonOtherPatientIDs ) 			      itk::EncapsulateMetaData<std::string>( dictionary, "0010|1000", "None"         ); // Other Patient IDs                           
      if ( ! flgDontAnonOtherPatientNames ) 			      itk::EncapsulateMetaData<std::string>( dictionary, "0010|1001", "None"         ); // Other Patient Names                         
      if ( ! flgDontAnonPatientsBirthName ) 			      itk::EncapsulateMetaData<std::string>( dictionary, "0010|1005", "Anonymous"    ); // Patient's Birth Name                        
      if ( ! flgDontAnonPatientsAge ) 				      itk::EncapsulateMetaData<std::string>( dictionary, "0010|1010", "0"            ); // Patient's Age                               
      if ( ! flgDontAnonPatientsSize ) 				      itk::EncapsulateMetaData<std::string>( dictionary, "0010|1020", "0"            ); // Patient's Size                              
      if ( ! flgDontAnonPatientsWeight ) 			      itk::EncapsulateMetaData<std::string>( dictionary, "0010|1030", "0"            ); // Patient's Weight                            
      if ( ! flgDontAnonPatientsAddress ) 			      itk::EncapsulateMetaData<std::string>( dictionary, "0010|1040", "None"         ); // Patient's Address                           
      if ( ! flgDontAnonPatientsMothersBirthName ) 		      itk::EncapsulateMetaData<std::string>( dictionary, "0010|1060", "Anonymous"    ); // Patient's Mother's Birth Name               
      if ( ! flgDontAnonPatientsTelephoneNumbers ) 		      itk::EncapsulateMetaData<std::string>( dictionary, "0010|2154", "None"         ); // Patient's Telephone Numbers                 
      if ( ! flgDontAnonAdditionalPatientHistory ) 		      itk::EncapsulateMetaData<std::string>( dictionary, "0010|21b0", "None"         ); // Additional Patient History                  
      if ( ! flgDontAnonPatientsReligiousPreference ) 		      itk::EncapsulateMetaData<std::string>( dictionary, "0010|21f0", "None"         ); // Patient's Religious Preference              
      if ( ! flgDontAnonPatientComments ) 			      itk::EncapsulateMetaData<std::string>( dictionary, "0010|4000", "None"         ); // Patient Comments                            
      if ( ! flgDontAnonPatientState ) 				      itk::EncapsulateMetaData<std::string>( dictionary, "0038|0500", "None"         ); // Patient State                               
      if ( ! flgDontAnonPatientTransportArrangements ) 		      itk::EncapsulateMetaData<std::string>( dictionary, "0040|1004", "None"         ); // Patient Transport Arrangements              
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

