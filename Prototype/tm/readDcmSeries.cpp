// ****************************************************************
//
//
//
//
// ****************************************************************

#include "itkOrientedImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

void writeVolumeAndHeader(itk::ImageSeriesReader< itk::OrientedImage< signed short, 3 > > * reader,  std::vector< std::string > & fileNames, const std::string & volName, const std::string & headerName, itk::MetaDataDictionary & dictionary);

int main( int argc, char* argv[] )
{

  if( argc < 2 )
    {
      std::cerr << "Usage: " << std::endl;
      std::cerr << argv[0] << " DicomDirectory  [seriesName]" 
		<< std::endl;
      return EXIT_FAILURE;
    }

  // We define the pixel type and dimension of the image to be read. 
  // We also choose to use the \doxygen{OrientedImage} in order to make sure
  // that the image orientation information contained in the direction cosines
  // of the DICOM header are read in and passed correctly down the image processing
  // pipeline.
  typedef signed short    PixelType;
  const unsigned int      Dimension = 3;

  typedef itk::OrientedImage< PixelType, Dimension >         ImageType;

  // We use the image type for instantiating the type of the series reader and
  // for constructing one object of its type.
  typedef itk::ImageSeriesReader< ImageType >        ReaderType;
  ReaderType::Pointer reader = ReaderType::New();


  // A GDCMImageIO object is created and connected to the reader. This object is
  // the one that is aware of the internal intricacies of the DICOM format. 
  typedef itk::GDCMImageIO       ImageIOType;
  ImageIOType::Pointer dicomIO = ImageIOType::New();
  
  reader->SetImageIO( dicomIO );

  // Now we face one of the main challenges of the process of reading a DICOM
  // series. That is, to identify from a given directory the set of filenames
  // that belong together to the same volumetric image. Fortunately for us, GDCM
  // offers functionalities for solving this problem and we just need to invoke
  // those functionalities through an ITK class that encapsulates a communication
  // with GDCM classes. This ITK object is the GDCMSeriesFileNames. Conveniently
  // for us, we only need to pass to this class the name of the directory where
  // the DICOM slices are stored. This is done with the \code{SetDirectory()}
  // method. The GDCMSeriesFileNames object will explore the directory and will
  // generate a sequence of filenames for DICOM files for one study/series. 
  // In this example, we also call the \code{SetUseSeriesDetails(true)} function
  // that tells the GDCMSereiesFileNames object to use additional DICOM 
  // information to distinguish unique volumes within the directory.  This is
  // useful, for example, if a DICOM device assigns the same SeriesID to 
  // a scout scan and its 3D volume; by using additional DICOM information
  // the scout scan will not be included as part of the 3D volume.  Note that
  // \code{SetUseSeriesDetails(true)} must be called prior to calling
  // \code{SetDirectory()}. By default \code{SetUseSeriesDetails(true)} will use
  // the following DICOM tags to sub-refine a set of files into multiple series:
  // * 0020 0011 Series Number
  // * 0018 0024 Sequence Name
  // * 0018 0050 Slice Thickness
  // * 0028 0010 Rows
  // * 0028 0011 Columns
  // If this is not enough for your specific case you can always add some more
  // restrictions using the \code{AddSeriesRestriction()} method. In this example we will use
  // the DICOM Tag: 0008 0021 DA 1 Series Date, to sub-refine each series. The format
  // for passing the argument is a string containing first the group then the element
  // of the DICOM tag, separed by a pipe (|) sign.

  typedef itk::GDCMSeriesFileNames NamesGeneratorType;
  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

  nameGenerator->SetUseSeriesDetails( true );
  //nameGenerator->AddSeriesRestriction("0008|0031" ); //"0008|0021" );

  nameGenerator->SetDirectory( argv[1] );

  std::cout << std::endl << "The directory: " << std::endl;
  std::cout << std::endl << argv[1] << std::endl << std::endl;
  std::cout << "Contains the following DICOM Series: ";
  std::cout << std::endl << std::endl;
   
  int Nvol = 0; //number of volumes found

  // The GDCMSeriesFileNames object first identifies the list of DICOM series
  // that are present in the given directory. We receive that list in a reference
  // to a container of strings and then we can do things like printing out all
  // the series identifiers that the generator had found. Since the process of
  // finding the series identifiers can potentially throw exceptions, it is
  // wise to put this code inside a try/catch block.
  typedef std::vector< std::string >    SeriesIdContainer;
    
  const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
    
  SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
  SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
  while( seriesItr != seriesEnd )
    {
      std::cout << seriesItr->c_str() << std::endl;
      seriesItr++;
      Nvol++;
    }
  std::cout << "The number of volumes found is " << Nvol << std::endl;

  // Given that it is common to find multiple DICOM series in the same directory,
  // we must tell the GDCM classes what specific series do we want to read. In
  // this example we do this by checking first if the user has provided a series
  // identifier in the command line arguments. If no series identifier has been
  // passed, then we simply use the first series found during the exploration of
  // the directory.
  std::string seriesIdentifier;

  typedef std::vector< std::string >   FileNamesContainer;
  FileNamesContainer fileNames;
    
  int i=1;

  if( argc > 3 ) // If no optional series identifier, then extract all
    {
      seriesIdentifier = argv[2];
    }
    else
      {
	seriesItr = seriesUID.begin();
	while( seriesItr != seriesEnd )
	  {
	    seriesIdentifier = seriesItr->c_str();
	    std::cout << std::endl << std::endl;
	    std::cout << "Now reading series: " << std::endl;
	    std::cout << seriesIdentifier << " ..." << std::endl;
	    // We pass the series identifier to the name generator and ask for all the
	    // filenames associated to that series. This list is returned in a container of
	    // strings by the GetFileNames() 
	    fileNames = nameGenerator->GetFileNames( seriesIdentifier );
	    //stringstream volnameTest; 
	    //volnameTest << "ser" << i << ".gipl";
	    //std::cout << "The name is "<< volnameTest.str() << std::endl;
	    stringstream volName, headerName; 
	    volName << seriesIdentifier << ".nii";//gipl";
	    headerName << seriesIdentifier << ".txt";
	    //volName << "ser" << i << ".nii";//.gipl";
	    //headerName << "ser" << i << ".txt";
	    //volName << "preContrast.gipl";
	    //headerName << "preContrast.txt";
	    //char volName[] = {'s','e','r','.','g','i','p','l', '\0'};
	    //char headerName[] =  {'s','e','r','.','t','x','t', '\0'};	    
	    writeVolumeAndHeader(reader, fileNames, volName.str(), headerName.str(), dicomIO->GetMetaDataDictionary());
	    seriesItr++;
	    i++;
	  }
      }

  return EXIT_SUCCESS;
}

void writeVolumeAndHeader(itk::ImageSeriesReader< itk::OrientedImage< signed short, 3 > > * reader,  std::vector< std::string > & fileNames, const std::string & volName, const std::string & headerName, itk::MetaDataDictionary & dictionary)
{

  // The list of filenames can now be passed to the reader
  reader->SetFileNames( fileNames );

  // Finally we can trigger the reading process
  try
    {
      reader->Update();
    }
  catch (itk::ExceptionObject &ex)
    {
      std::cout << ex << std::endl;
    }

  // Extract the header information of this volume
  typedef itk::MetaDataDictionary   DictionaryType;
  //const  DictionaryType & dictionary = dicomIO->GetMetaDataDictionary();
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  DictionaryType::ConstIterator itr = dictionary.Begin();
  DictionaryType::ConstIterator end = dictionary.End();
 
  ofstream myFile;
  myFile.open( headerName.c_str() );

  while( itr != end )
    {
      itk::MetaDataObjectBase::Pointer  entry = itr->second;

      MetaDataStringType::Pointer entryvalue = 	dynamic_cast<MetaDataStringType *>( entry.GetPointer() );

      if( entryvalue )
	{
	  std::string tagkey   = itr->first;
	  std::string tagvalue = entryvalue->GetMetaDataObjectValue();
	  //std::cout << tagkey <<  " = " << tagvalue << std::endl;
	  myFile << tagkey <<  " = " << tagvalue << std::endl;
	}

      ++itr;
    }

  myFile.close();

  typedef itk::ImageFileWriter< itk::OrientedImage< signed short, 3 >  > WriterType;
  WriterType::Pointer writer = WriterType::New();
    
  writer->SetFileName( volName );

  writer->SetInput( reader->GetOutput() );

  std::cout  << "Writing the image as " <<  volName << std::endl;

  // Write the volume
  try
    {
      writer->Update();
    }
  catch (itk::ExceptionObject &ex)
    {
      std::cout << ex << std::endl;
    }

}



// this part is to print a specific tag of the volume
// check also the cpp file in backups
//   std::string entryId = "0008|0032";

//   DictionaryType::ConstIterator tagItr = dictionary.Find( entryId );

//   if( tagItr == end )
//     {
//     std::cerr << "Tag " << entryId;
//     std::cerr << " not found in the DICOM header" << std::endl;
//     return EXIT_FAILURE;
//     }

//   MetaDataStringType::ConstPointer entryvalue = 
//     dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

//   if( entryvalue )
//     {
//     std::string tagvalue = entryvalue->GetMetaDataObjectValue();
//     std::cout << "Now reading series: " << std::endl << std::endl;
//     std::cout << seriesIdentifier << std::endl;
//     std::cout << "Which has an Acquisition Time (" << entryId <<  ") ";
//     std::cout << " that is: " << tagvalue << std::endl;
//     }
//   else
//     {
//     std::cerr << "Entry was not of string type" << std::endl;
//     return EXIT_FAILURE;
//     }
