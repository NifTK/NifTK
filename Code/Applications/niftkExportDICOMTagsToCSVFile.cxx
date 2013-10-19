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


#include <fstream>
#include <iomanip>

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

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>

#include <vector>

#include <niftkExportDICOMTagsToCSVFileCLP.h>


namespace fs = boost::filesystem;

typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;



struct arguments
{
  std::string dcmDirectoryIn;
  std::string fileInputTagKeys;  
  std::string fileOutputCSV;  

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
// DoMain()
// -------------------------------------------------------------------------

template <unsigned int InputDimension, class OutputPixelType>
int DoMain(arguments args, std::vector<std::string> &tagList, std::fstream &foutTagsCSV)
{
  float progress = 0.;
  float iFile = 0.;
  float nFiles;
 
  itksys_ios::ostringstream value;


  std::string fileInputFullPath;
  std::string fileInputRelativePath;
  std::string fileOutputRelativePath;
  std::string fileOutputFullPath;
  std::string dirOutputFullPath;
    
  typedef float InternalPixelType;

  typedef itk::Image< InternalPixelType, InputDimension > InternalImageType; 

  typedef itk::ImageFileReader< InternalImageType > ReaderType;

  typename ReaderType::Pointer reader = ReaderType::New();
  typename InternalImageType::Pointer image;

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
    std::cout << "WARNING: Skipping file (not DICOM?): " << args.iterFilename << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "File: " << args.iterFilename << std::endl;


  image = reader->GetOutput();
  image->DisconnectPipeline();

  DictionaryType &dictionary = image->GetMetaDataDictionary();

  unsigned int iTag = 0;
  std::vector< std::string >::iterator iterTags;     

  foutTagsCSV << boost::filesystem::canonical( args.iterFilename ) << ",";

  // Get each tag

  for ( iterTags = tagList.begin(); 
	iterTags < tagList.end(); 
	++iterTags, iTag += 1. )
  {
  
    DictionaryType::ConstIterator tagItr;
    DictionaryType::ConstIterator tagEnd;

    std::string tagID;
    std::string tagValue;

    tagItr = dictionary.Find( *iterTags );
    tagEnd = dictionary.End();
   
    if( tagItr != tagEnd )
    {
      MetaDataStringType::ConstPointer entryvalue = 
        dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

      if ( entryvalue )
      {
        tagValue = entryvalue->GetMetaDataObjectValue();

        bool found =  itk::GDCMImageIO::GetLabelFromTag( *iterTags, tagID );

        if ( args.flgVerbose )
        {
          std::cout << std::setw(12) << iTag << " Tag (" << *iterTags <<  ") " << tagID
                    << " is: " << tagValue << std::endl;
        }

        if ( found )
        {
          foutTagsCSV << ",\"" << tagValue << "\"";
        }
        else
        {
          foutTagsCSV << ",";
        }
      }
    }
  }

  foutTagsCSV << std::endl;
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


  if ( fileInputTagKeys.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    std::cerr << "ERROR: The input tag file must be specified" << std::endl;
    return EXIT_FAILURE;
  }

  args.dcmDirectoryIn   = dcmDirectoryIn;                     
  args.fileInputTagKeys = fileInputTagKeys;                    
  args.fileOutputCSV    = fileOutputCSV;                    

  args.flgVerbose       = flgVerbose;    


  std::cout << std::endl << "Examining directory: " 
	    << args.dcmDirectoryIn << std::endl << std::endl;


  // Open the output csv file
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  std::fstream foutTagsCSV;

  foutTagsCSV.open( fileOutputCSV.c_str(), std::ios::out );

  if ((! foutTagsCSV) || foutTagsCSV.bad()) {
    std::cerr << "ERROR: Failed to open file: " << fileOutputCSV.c_str() << std::endl;
    return EXIT_FAILURE;   
  }

  foutTagsCSV << "\"File Name\"";


  // Read the list of tags
  // ~~~~~~~~~~~~~~~~~~~~~

  unsigned int nTags = 0;
  std::string tagID;

  std::vector<std::string> tagList;

  std::fstream finTagKeys;

  finTagKeys.open( fileInputTagKeys.c_str(), std::ios::in );

  if ((! finTagKeys) || finTagKeys.bad()) {
    std::cerr << "ERROR: Failed to open file: " << fileInputTagKeys.c_str() << std::endl;
    return EXIT_FAILURE;   
  }

  if ( finTagKeys.is_open() )                  
  {                                      
    while( !finTagKeys.eof() )
    {                                                                                   
      std::string tmp;                                                                          
      finTagKeys >> tmp;                                                                    

      if ( tmp.length() > 0 )
      {
        if ( itk::GDCMImageIO::GetLabelFromTag( tmp, tagID ) )
        {
      
          tagList.push_back( tmp );                                                          
          nTags++;

          std::cout << std::setw(12) << nTags << ": " 
                    << tmp << " " << tagID << std::endl;

          foutTagsCSV << ",\"" << tmp << " " << tagID << "\"";
        }
        else 
        {
          std::cout << std::setw(12) << nTags << ": ERROR - TAG NOT FOUND SO IGNORED: " 
                    << tmp << " " << tagID << std::endl;
        }
      }
    }                                                                                   
  }                                                                                     

  foutTagsCSV << std::endl;

  if ( nTags == 0 )                      
  {                                                                                       
    std::cerr << "ERROR: No tags detected" << std::endl;
    finTagKeys.close();
    return EXIT_FAILURE;
  }  

  finTagKeys.close();


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

    try
    {
      imageIO = itk::ImageIOFactory::CreateImageIO(args.iterFilename.c_str(), 
                                                   itk::ImageIOFactory::ReadMode);

      if ( ( ! imageIO ) || ( ! imageIO->CanReadFile( args.iterFilename.c_str() ) ) )
      {
        std::cerr << "WARNING: Failed to read DICOM tags, skipping file: " 
                  << args.iterFilename << std::endl;
        continue;
      }


      unsigned int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.iterFilename);

      if (dims != 3 && dims != 2)
      {
        std::cout << "WARNING: Unsupported image dimension (" << dims << ") for file: " 
                  << args.iterFilename << std::endl;
        continue;
      }


      int result;

      switch ( dims )
      {
      case 2:
      {
        switch (itk::PeekAtComponentType(args.iterFilename))
        {
        case itk::ImageIOBase::UCHAR:
          result = DoMain<2, unsigned char>(args, tagList, foutTagsCSV);  
          break;
    
        case itk::ImageIOBase::CHAR:
          result = DoMain<2, char>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::USHORT:
          result = DoMain<2, unsigned short>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::SHORT:
          result = DoMain<2, short>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::UINT:
          result = DoMain<2, unsigned int>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::INT:
          result = DoMain<2, int>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::ULONG:
          result = DoMain<2, unsigned long>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::LONG:
          result = DoMain<2, long>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::FLOAT:
          result = DoMain<2, float>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::DOUBLE:
          result = DoMain<2, double>(args, tagList, foutTagsCSV);  
          break;

        default:
          std::cerr << "WARNING: Unrecognised pixel type, skipping file: " 
                    << args.iterFilename << std::endl;
        }

        break;
      }

      case 3:
      {
        switch (itk::PeekAtComponentType(args.iterFilename))
        {
        case itk::ImageIOBase::UCHAR:
          result = DoMain<3, unsigned char>(args, tagList, foutTagsCSV);  
          break;
    
        case itk::ImageIOBase::CHAR:
          result = DoMain<3, char>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::USHORT:
          result = DoMain<3, unsigned short>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::SHORT:
          result = DoMain<3, short>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::UINT:
          result = DoMain<3, unsigned int>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::INT:
          result = DoMain<3, int>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::ULONG:
          result = DoMain<3, unsigned long>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::LONG:
          result = DoMain<3, long>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::FLOAT:
          result = DoMain<3, float>(args, tagList, foutTagsCSV);  
          break;

        case itk::ImageIOBase::DOUBLE:
          result = DoMain<3, double>(args, tagList, foutTagsCSV);  
          break;

        default:
          std::cerr << "WARNING: Unrecognised pixel type, skipping file: " 
                    << args.iterFilename << std::endl;
        }

        break;
      }

      default:
      {
        std::cout << "WARNING: Unsupported image dimension (" << dims << ") for file: " 
                  << args.iterFilename << std::endl;
      }
      }

      std::cout << std::endl;
    }

    catch (itk::ExceptionObject &ex)
    {
      std::cerr << "WARNING: Failed to read DICOM tags, skipping file: " 
		<< args.iterFilename << std::endl;
      continue;
    }
  }

  foutTagsCSV.close();

  return EXIT_SUCCESS;
}
 
 

