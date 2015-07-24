/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

/*!
 * \file niftkPrintDICOMSeries.cxx 
 * \page niftkPrintDICOMSeries
 * \section niftkPrintDICOMSeriesSummary niftkPrintDICOMSeries
 * 
 * Search for DICOM mammograms in a directory and anonymise them by removing patient information from the DICOM header and/or applying a rectangular mask to remove the label.
 *
 */


#include <niftkFileHelper.h>
#include <niftkConversionUtils.h>

#include <itkCommandLineHelper.h>
#include <itkLogHelper.h>

#include <itkImageSeriesReader.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <vector>

#include <niftkPrintDICOMSeriesCLP.h>


namespace fs = boost::filesystem;

typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;



// -----------------------------------------------------------------------------
// PrintTag()
// -----------------------------------------------------------------------------

void PrintTag( std::fstream *fout,
               const DictionaryType &dictionary, 
               std::string entryId )
{
  DictionaryType::ConstIterator tagItr;
  DictionaryType::ConstIterator end = dictionary.End();

  //  It is also possible to read a specific tag. In that case the string of the
  //  entry can be used for querying the MetaDataDictionary.

  tagItr = dictionary.Find( entryId );

  // If the entry is actually found in the Dictionary, then we can attempt to
  // convert it to a string entry by using a \code{dynamic\_cast}.

  if ( tagItr != end )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

    // If the dynamic cast succeed, then we can print out the values of the label,
    // the tag and the actual value.
	
    if ( entryvalue )
    {
      std::string tagkey = tagItr->first;
      std::string tagID;
      bool found =  itk::GDCMImageIO::GetLabelFromTag( tagkey, tagID );

      std::string tagValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << "   " << tagID <<  ": " << tagValue << std::endl;

      if ( fout )
      {
        *fout << "   " << tagID <<  ": " << tagValue << std::endl;
      }
    }
  }
}


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  std::fstream *fout = 0;

  typedef signed short    PixelType;
  const unsigned int      Dimension = 3;

  typedef itk::Image< PixelType, Dimension > ImageType;
  typedef itk::ImageFileReader< ImageType > ReaderType;


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  std::cout << "Input DICOM directory: " << dcmDirectoryIn << std::endl
            << "Output text file: " << fileOutputText << std::endl << std::endl;


  if ( dcmDirectoryIn.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    std::cerr << "ERROR: The input directory must be specified" << std::endl;
    return EXIT_FAILURE;
  }

  if ( fileOutputText.length() != 0 )
  {
    fout = new std::fstream( fileOutputText.c_str(), std::ios::out );
    
    if ((! *fout) || fout->bad()) 
    {
      std::cerr << "ERROR: Failed to open file: " << fileOutputText << std::endl;
      exit( EXIT_FAILURE );
    }
  }
    


  // Get the DICOM Series

  std::vector< std::string > fileNames;
  std::vector< std::string >::iterator iterFilenames;
     
  typedef itk::GDCMSeriesFileNames NamesGeneratorType;

  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

  nameGenerator->SetLoadSequences( true );
  nameGenerator->SetLoadPrivateTags( true ); 

  nameGenerator->SetUseSeriesDetails( true );

  nameGenerator->SetDirectory( dcmDirectoryIn );
  
  try
  {      
    
    // The GDCMSeriesFileNames object first identifies the list of DICOM series
    // that are present in the given directory. We receive that list in a
    // reference to a container of strings and then we can do things like
    // printing out all the series identifiers that the generator had
    // found. Since the process of finding the series identifiers can
    // potentially throw exceptions, it is wise to put this code inside a
    // try/catch block.

    typedef std::vector<std::string> seriesIdContainer;
    const seriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

    seriesIdContainer::const_iterator seriesItr = seriesUID.begin();
    seriesIdContainer::const_iterator seriesEnd = seriesUID.end();

    if ( flgVerbose ) 
    {
      std::cout << std::endl << "The directory: " << std::endl
                << std::endl << dcmDirectoryIn << std::endl << std::endl
                << "Contains the following DICOM Series: "
                << std::endl << std::endl;
    }

    if ( fout )
    {
      *fout << std::endl << "The directory: " << std::endl
            << std::endl << dcmDirectoryIn << std::endl << std::endl
            << "Contains the following DICOM Series: "
            << std::endl << std::endl;
    }

    while( seriesItr != seriesEnd )
    {
      std::cout << std::endl << "Series: " << seriesItr->c_str() << std::endl;

      if ( fout )
      {
        *fout << std::endl << "Series: " << seriesItr->c_str() << std::endl;
      }

      fileNames = nameGenerator->GetFileNames( *seriesItr );
      
      
      // Read the first image in this series


      typedef itk::ImageFileReader< ImageType > ReaderType;

      ReaderType::Pointer reader = ReaderType::New();
      reader->SetFileName( fileNames[0] );
      
      typedef itk::GDCMImageIO ImageIOType;

      ImageIOType::Pointer gdcmImageIO = ImageIOType::New();

      gdcmImageIO->SetMaxSizeLoadEntry(0xffff);

      reader->SetImageIO( gdcmImageIO );

      try
      {
        reader->Update();
      }
      catch ( itk::ExceptionObject &e )
      {
        std::cerr << "ERROR: Failed to read file: " << fileNames[0] << std::endl;
        std::cerr << e << std::endl;
        if ( fout ) fout->close();
        exit( EXIT_FAILURE );
      }
 
      const  DictionaryType &dictionary = gdcmImageIO->GetMetaDataDictionary();

      
      PrintTag( fout, dictionary, "0010|0010" ); // Patient name
      PrintTag( fout, dictionary, "0008|0060" ); // Modality
      PrintTag( fout, dictionary, "0008|0021" ); // Series date
      PrintTag( fout, dictionary, "0008|0031" ); // Series time
      PrintTag( fout, dictionary, "0008|103e" ); // Series description
      PrintTag( fout, dictionary, "0018|1030" ); // Protocol name
      PrintTag( fout, dictionary, "0018|0015" ); // Body part
      PrintTag( fout, dictionary, "0020|0011" ); // Series number

      // Print all the images in this series

      for (iterFilenames=fileNames.begin(); iterFilenames<fileNames.end(); ++iterFilenames) 
      {
        std::cout << "      " << *iterFilenames << std::endl;        

        if ( fout )
        {
          *fout << "      " << *iterFilenames << std::endl;        
        }
      }

      seriesItr++;
    }
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cout << ex << std::endl;

    if ( fout ) fout->close();
    return EXIT_FAILURE;
  }

  if ( fout )
  {
    fout->close();
    delete fout;
  }
 
  return EXIT_SUCCESS;
}
 
 

