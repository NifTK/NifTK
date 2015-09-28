/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

/*!
 * \file niftkAnonymiseDICOMImages.cxx
 * \page niftkAnonymiseDICOMImages
 * \section niftkAnonymiseDICOMImagesSummary niftkAnonymiseDICOMImages
 *
 * Search for DICOM images in a directory and anonymise them by removing patient information from the DICOM header.
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

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <vector>

#include <niftkAnonymiseDICOMImagesCLP.h>


namespace fs = boost::filesystem;

typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;


struct arguments
{
  std::string dcmDirectoryIn;
  std::string dcmDirectoryOut;
  std::string strAdd2Suffix;

  bool flgOverwrite;
  bool flgVerbose;

  bool flgDontAnonPatientsName;
  std::string strPatientsName;

  bool flgDontAnonPatientsBirthDate;
  std::string strPatientsBirthDate;

  bool flgDontAnonOtherPatientNames;
  std::string strOtherPatientNames;

  bool flgDontAnonPatientsBirthName;
  std::string strPatientsBirthName;

  bool flgDontAnonPatientsAddress;
  std::string strPatientsAddress;

  bool flgDontAnonPatientsMothersBirthName;
  std::string strPatientsMothersBirthName;

  bool flgDontAnonPatientsTelephoneNumbers;
  std::string strPatientsTelephoneNumbers;


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

  DictionaryType::ConstIterator tagItr = dictionary.Find( tagID );
  DictionaryType::ConstIterator end = dictionary.End();

  if ( tagItr != end )
  {
    MetaDataStringType::ConstPointer entryvalue =
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

    if ( entryvalue )
    {
      std::string tagValue = entryvalue->GetMetaDataObjectValue();

      std::cout << "Anonymising tag (" << tagID <<  ") "
		<< " from: " << tagValue
		<< " to: " << newTagValue << std::endl;

      itk::EncapsulateMetaData<std::string>( dictionary, tagID, newTagValue );
    }
  }

};


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

template <class InputPixelType>
int DoMain(arguments args, InputPixelType min, InputPixelType max)
{

  // Iterate through each file and anonymise it
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

  image = reader->GetOutput();
  image->DisconnectPipeline();


  // Anonymise the DICOM header

  AnonymiseTag( args.flgDontAnonPatientsName,  	    dictionary, "0010|0010", "Anonymous"    ); // Patient's Name
  AnonymiseTag( args.flgDontAnonPatientsBirthDate,	    dictionary, "0010|0030", "00000000"     ); // Patient's Birth Date
  AnonymiseTag( args.flgDontAnonOtherPatientNames, 	    dictionary, "0010|1001", "None"         ); // Other Patient Names
  AnonymiseTag( args.flgDontAnonPatientsBirthName, 	    dictionary, "0010|1005", "Anonymous"    ); // Patient's Birth Name
  AnonymiseTag( args.flgDontAnonPatientsAddress, 	    dictionary, "0010|1040", "None"         ); // Patient's Address
  AnonymiseTag( args.flgDontAnonPatientsMothersBirthName, dictionary, "0010|1060", "Anonymous"    ); // Patient's Mother's Birth Name
  AnonymiseTag( args.flgDontAnonPatientsTelephoneNumbers, dictionary, "0010|2154", "None"         ); // Patient's Telephone Numbers


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

  args.flgOverwrite            = flgOverwrite;
  args.flgVerbose              = flgVerbose;

  args.flgDontAnonPatientsName = flgDontAnonPatientsName;
  args.strPatientsName         = strPatientsName;

  args.flgDontAnonPatientsBirthDate = flgDontAnonPatientsBirthDate;
  args.strPatientsBirthDate	    = strPatientsBirthDate;

  args.flgDontAnonOtherPatientNames = flgDontAnonOtherPatientNames;
  args.strOtherPatientNames	    = strOtherPatientNames;

  args.flgDontAnonPatientsBirthName = flgDontAnonPatientsBirthName;
  args.strPatientsBirthName	    = strPatientsBirthName;

  args.flgDontAnonPatientsAddress = flgDontAnonPatientsAddress;
  args.strPatientsAddress	  = strPatientsAddress;

  args.flgDontAnonPatientsMothersBirthName = flgDontAnonPatientsMothersBirthName;
  args.strPatientsMothersBirthName         = strPatientsMothersBirthName;

  args.flgDontAnonPatientsTelephoneNumbers = flgDontAnonPatientsTelephoneNumbers;
  args.strPatientsTelephoneNumbers         = strPatientsTelephoneNumbers;


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



