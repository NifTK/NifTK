/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <ConversionUtils.h>
#include <CommandLineParser.h>

#include <itkOrientedImage.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkImageFileWriter.h>

#include <vector>

/*!
 * \file niftkDicomSeriesReadImageWrite.cxx
 * \page niftkDicomSeriesReadImageWrite
 * \section niftkDicomSeriesReadImageWriteSummary Reads DICOM series using ITK and hence GDCM, and writes an image volume using ITK.
 */

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_STRING, "series", "name", "The input series name required."},
  {OPT_STRING, "filter", "pattern", "Only consider DICOM files that contain the string 'pattern'."},

  {OPT_STRING|OPT_REQ, "of", "filestem", "The output image volume(s) filestem (filename will be: 'filestem_%2d.suffix')."},
  {OPT_STRING|OPT_REQ, "os", "suffix",   "The output image suffix to use when using option '-or'."},

  {OPT_STRING|OPT_REQ|OPT_LONELY, NULL, "directory", "Input DICOM directory."},

  {OPT_DONE, NULL, NULL, 
   "Program to convert the content of a DICOM directory into image volumes.\n"
  }
};

enum { 
  O_VERBOSE,
  O_DEBUG,

  O_SERIES,
  O_FILENAME_FILTER,

  O_OUTPUT_FILESTEM,
  O_OUTPUT_FILESUFFIX,

  O_INPUT_DICOM_DIRECTORY
};


struct arguments
{
  // Set up defaults
  arguments()
  {
    flgVerbose = false;
    flgDebug = false;

    fileOutputStem = 0;
    fileOutputSuffix = 0;
  }

  bool flgVerbose;
  bool flgDebug;
  
  std::string seriesName;
  std::string fileNameFilter;

  char *fileOutputStem;
  char *fileOutputSuffix;

  std::string dirDICOMInput;
};
  

// We define the pixel type and dimension of the image to be read. In this
// particular case, the dimensionality of the image is 3, and we assume a
// \code{signed short} pixel type that is commonly used for X-Rays CT scanners.
// 
// We also choose to use the \doxygen{OrientedImage} in order to make sure
// that the image orientation information contained in the direction cosines
// of the DICOM header are read in and passed correctly down the image processing
// pipeline.

typedef signed short    PixelType;
const unsigned int      Dimension = 3;

typedef itk::OrientedImage< PixelType, Dimension > ImageType;

// We use the image type for instantiating the type of the series reader and
// for constructing one object of its type.

typedef itk::ImageSeriesReader< ImageType > ReaderType;

typedef itk::GDCMSeriesFileNames NamesGeneratorType;



// -----------------------------------------------------------------------------
// WriteSeriesAsVolume()
// -----------------------------------------------------------------------------

bool WriteSeriesAsVolume( bool flgVerbose,
			  bool flgDebug,
			  std::string seriesIdentifier,
			  std::string filenameFilter,
			  char *fileOutput,
			  NamesGeneratorType *nameGenerator,
			  ReaderType *reader ) 
{
  std::vector< std::string >::iterator iterFilenames;

  // We pass the series identifier to the name generator and ask for all the
  // filenames associated to that series. This list is returned in a container of
  // strings by the \code{GetFileNames()} method. 
  
  typedef std::vector< std::string > FileNamesContainer;
  FileNamesContainer fileNames;
  
  fileNames = nameGenerator->GetFileNames( seriesIdentifier );

  // If the user has specified a filename substring then
  // we can use this to filter out the images we don't want

  if ( filenameFilter.length() > 0 ) {

    FileNamesContainer filteredFileNames;
    
    for (iterFilenames=fileNames.begin(); iterFilenames<fileNames.end(); ++iterFilenames) 

      if ( iterFilenames->find( filenameFilter ) != std::string::npos )
	filteredFileNames.push_back( *iterFilenames );

      else if ( flgDebug ) 
	std::cout << "   ignoring file: " << *iterFilenames << std::endl;


    if ( filteredFileNames.size() < 1 )
      return false;
    else
      fileNames = filteredFileNames;
  }


  if ( flgVerbose ) {

    std::cout << std::endl << std::endl;
    std::cout << "Now reading series: " << std::endl << std::endl;
    std::cout << seriesIdentifier << std::endl;
    std::cout << std::endl << std::endl;

   for (iterFilenames=fileNames.begin(); iterFilenames<fileNames.end(); ++iterFilenames) 
      std::cout << "   files: " << *iterFilenames << std::endl;
  }


  // The list of filenames can now be passed to the \doxygen{ImageSeriesReader}
  // using the \code{SetFileNames()} method.
  //  
  //  \index{itk::ImageSeriesReader!SetFileNames()}
  
  reader->SetFileNames( fileNames );

  // Finally we can trigger the reading process by invoking the \code{Update()}
  // method in the reader. This call as usual is placed inside a \code{try/catch}
  // block.

  try
    {
      reader->UpdateLargestPossibleRegion();
    }
  catch (itk::ExceptionObject &ex)
    {
      std::cout << ex << std::endl;
      return false;
    }

  // At this point, we have a volumetric image in memory that we can access by
  // invoking the \code{GetOutput()} method of the reader.
  //
  // We proceed now to save the volumetric image in another file, as specified by
  // the user in the command line arguments of this program. Thanks to the
  // ImageIO factory mechanism, only the filename extension is needed to identify
  // the file format in this case.

  typedef itk::ImageFileWriter< ImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
    
  writer->SetFileName( fileOutput );

  writer->SetInput( reader->GetOutput() );

  std::cout  << "Writing the image as " << std::endl << std::endl;
  std::cout  << fileOutput << std::endl << std::endl;

  // The process of writing the image is initiated by invoking the
  // \code{Update()} method of the writer.

  try
    {
      writer->Update();
    }
  catch (itk::ExceptionObject &ex)
    {
      std::cout << ex << std::endl;
      return false;
    }

  return true;
}



// -----------------------------------------------------------------------------
// main()
// -----------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  char fileOutput[512];


  // To pass around command line args
  struct arguments args;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );

  CommandLineOptions.GetArgument( O_SERIES, args.seriesName );
  CommandLineOptions.GetArgument( O_FILENAME_FILTER, args.fileNameFilter );

  CommandLineOptions.GetArgument( O_OUTPUT_FILESTEM, args.fileOutputStem );
  CommandLineOptions.GetArgument( O_OUTPUT_FILESUFFIX, args.fileOutputSuffix );

  CommandLineOptions.GetArgument( O_INPUT_DICOM_DIRECTORY, args.dirDICOMInput );
  

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
  // that tells the GDCMSeriesFileNames object to use additional DICOM 
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

  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

  nameGenerator->SetUseSeriesDetails( true );
  nameGenerator->AddSeriesRestriction("0008|0021" );

  nameGenerator->SetDirectory( args.dirDICOMInput );
  
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
      
      if ( args.flgVerbose ) {
	std::cout << std::endl << "The directory: " << std::endl;
	std::cout << std::endl << args.dirDICOMInput << std::endl << std::endl;
	std::cout << "Contains the following DICOM Series: ";
	std::cout << std::endl << std::endl;
      }

      while( seriesItr != seriesEnd )
	{
	  std::cout << seriesItr->c_str() << std::endl;
	  seriesItr++;
	}
      
      // Given that it is common to find multiple DICOM series in the same directory,
      // we must tell the GDCM classes what specific series do we want to read. In
      // this example we do this by checking first if the user has provided a series
      // identifier in the command line arguments. 
      
      if( args.seriesName.length() > 0 )
	{
	  sprintf( fileOutput, "%s.%s", args.fileOutputStem, args.fileOutputSuffix );
	  
	  WriteSeriesAsVolume( args.flgVerbose, 
			       args.flgDebug, 
			       args.seriesName, 
			       args.fileNameFilter,
			       fileOutput, 
			       nameGenerator, 
			       reader );
	}
      
      // Otherwise we output all the images

      else
	{
	
	  seriesItr = seriesUID.begin();
	  seriesEnd = seriesUID.end();

	  int i = 0;
	  while( seriesItr != seriesEnd )
	    {
	      sprintf( fileOutput, "%s_%02d.%s", args.fileOutputStem, i, args.fileOutputSuffix );

	      if ( WriteSeriesAsVolume( args.flgVerbose, 
					args.flgDebug, 
					*seriesItr, 
					args.fileNameFilter,
					fileOutput, 
					nameGenerator, 
					reader ) )
		i++;

	      seriesItr++;
	    }
	}

    }
  catch (itk::ExceptionObject &ex)
    {
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }
  
  return EXIT_SUCCESS;
}

