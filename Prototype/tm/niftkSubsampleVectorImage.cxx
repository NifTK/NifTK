
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh                 $
 $Date:: 2011-12-16 13:12:13 +#$
 $Rev:: 8041                   $

 Copyright (c) UCL : See the file NifTKCopyright.txt in the top level 
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <math.h>
#include <float.h>

#include "LogHelper.h"
#include "ConversionUtils.h"
#include "CommandLineParser.h"
#include "itkCommandLineHelper.h"

#include "itkSubsampleImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkVector.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_DOUBLE, "sub", "value", "The subsampling factor (greater than 1) [1]."},

  {OPT_STRING, "o", "filename", "The output sub-sampled image."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to subsample an image by a certain factor and apply the "
   "appropriate blurring (equivalent to voxel averaging for integer "
   "subsampling factors)."
  }
};


enum {
  O_VERBOSE,
  O_DEBUG,

  O_SUBSAMPLING_FACTOR,

  O_OUTPUT_IMAGE,
  O_INPUT_IMAGE
};



struct arguments
{
  bool flgVerbose;
  bool flgDebug;

  double factor;

  std::string fileOutputImage;
  std::string fileInputImage;
};


template <int Dimension>
int DoMain(arguments args)
{
  unsigned int i;
  
  double factors[ Dimension ];

  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  // Define the dimension of the images
  typedef float InputPixelType; 

  typedef itk::Vector< InputPixelType, Dimension > VectorType;
  
  typedef itk::Image<VectorType, Dimension> InputImageType;

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  typename FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName( args.fileInputImage );

  try
  { 
    niftk::LogHelper::InfoMessage("Reading the input image");
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Create the subsampling filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef float OutputPixelType;
  typedef itk::Image<VectorType, Dimension> OutputImageType;

  typedef itk::SubsampleImageFilter< InputImageType, OutputImageType > SubsampleImageFilterType;

  typename SubsampleImageFilterType::Pointer sampler = SubsampleImageFilterType::New();

  for (i=0; i<Dimension; i++)
    factors[i] = args.factor;

  sampler->SetSubsamplingFactors( factors );

  sampler->SetInput( imageReader->GetOutput() );
  
  try
  {
    niftk::LogHelper::InfoMessage("Computing subsampled image");
    sampler->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }


  // Write the subsampled image?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutputImage.length() != 0 ) {

    typedef itk::ImageFileWriter< OutputImageType > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( args.fileOutputImage );
    writer->SetInput( sampler->GetOutput() );

    try
      {
	niftk::LogHelper::InfoMessage("Writing the output image.");
	writer->Update();
      }
    catch (itk::ExceptionObject &e)
      {
	std::cerr << e << std::endl;
	return EXIT_FAILURE;
      }
  }
  
  return EXIT_SUCCESS;
}

int main( int argc, char *argv[] )
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.factor = 1.0;

  log4cplus::LogLevel logLevel = log4cplus::INFO_LOG_LEVEL;


  // This reads logging configuration from log4cplus.properties
  niftk::LogHelper::SetupBasicLogging();
  niftk::LogHelper::SetLogLevel(logLevel);
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  if (CommandLineOptions.GetArgument(O_DEBUG, args.flgDebug)) {
    logLevel = log4cplus::DEBUG_LOG_LEVEL;
    niftk::LogHelper::SetLogLevel(logLevel);
  }

  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );
  CommandLineOptions.GetArgument( O_DEBUG, args.flgDebug );

  CommandLineOptions.GetArgument( O_SUBSAMPLING_FACTOR, args.factor );

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, args.fileOutputImage );
  CommandLineOptions.GetArgument( O_INPUT_IMAGE, args.fileInputImage );


  unsigned int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fileInputImage);

  if (dims != 3 && dims != 2)
    {
      std::cout << "Unsupported image dimension" << std::endl;
      return EXIT_FAILURE;
    }

  int result;

  switch ( dims )
    {
      case 2:
        niftk::LogHelper::DebugMessage("Images are 2D");
        result = DoMain<2>(args);
        break;
      case 3:
        niftk::LogHelper::DebugMessage("Images are 3D");
        result = DoMain<3>(args);
      break;
      default:
        std::cout << "Unsuported image dimension" << std::endl;
        exit( EXIT_FAILURE );
    }
  return result;
}
