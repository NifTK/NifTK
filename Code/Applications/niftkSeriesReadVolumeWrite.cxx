/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkOrientedImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"

#include <vector>

/*!
 * \file niftkSeriesReadVolumeWrite.cxx
 * \page niftkSeriesReadVolumeWrite
 * \section niftkSeriesReadVolumeWriteSummary Reads a series of images and write an image volume using ITK.
 */

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_STRING|OPT_REQ, "o", "fileOut", "The output image volume filename."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "images", "Image filenames to be combined."},
  
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, 
   "Program to convert a set of images into a an image volume.\n"
  }
};

enum { 
  O_VERBOSE,
  O_DEBUG,

  O_OUTPUT_FILE,

  O_INPUT_IMAGES,
  O_MORE
};



// -----------------------------------------------------------------------------
// main()
// -----------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  char *fileInput = 0;	// A mandatory character string argument
  char **filesIn = 0;

  int i;			// Loop counter
  int arg;			// Index of arguments in command line 
  int nFiles = 0;

  bool flgVerbose = false;
  bool flgDebug = false;
  
  char *fileOutput = 0;

  std::vector<std::string> filenames;

  typedef signed short    PixelType;
  const unsigned int      Dimension = 3;

  typedef itk::OrientedImage< PixelType, Dimension > ImageType;

  typedef itk::ImageSeriesReader< ImageType > ReaderType;


  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, flgVerbose );
  CommandLineOptions.GetArgument( O_DEBUG, flgDebug );

  CommandLineOptions.GetArgument( O_OUTPUT_FILE, fileOutput );

  CommandLineOptions.GetArgument(O_INPUT_IMAGES, fileInput );

  // Call the 'OPT_MORE' option to determine the position of the list
  // of extra command line options ('arg').
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc) {            // Many strings
    nFiles = argc - arg + 1;
    filesIn = &argv[arg-1];

    std::cout << std::endl << "Input strings: " << std::endl;
    for (i=0; i<nFiles; i++) {
      filenames.push_back( std::string( filesIn[i] ) );
      std::cout << "   " << i+1 << " " << filesIn[i] << std::endl;
    }
  }
  else if (fileInput) {	// Single string
    nFiles = 1;
    filenames.push_back( std::string( fileInput ) );

    std::cout << std::endl << "Input string: " << fileInput << std::endl;
  }
  else {
    std::cerr << "ERROR: No files specified." << std::endl;
    return EXIT_FAILURE;
  }


  // Read the input images

  ReaderType::Pointer reader = ReaderType::New();
 
  reader->SetFileNames( filenames );

  
  try
  {
    reader->UpdateLargestPossibleRegion();
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cout << ex << std::endl;
    return false;
  }



  // and write them as a volume

  typedef itk::ImageFileWriter< ImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
    
  writer->SetFileName( fileOutput );

  writer->SetInput( reader->GetOutput() );

  std::cout  << "Writing the image as " << std::endl << std::endl;
  std::cout  << fileOutput << std::endl << std::endl;

  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cout << ex << std::endl;
    return false;
  }


  return EXIT_SUCCESS;
}

