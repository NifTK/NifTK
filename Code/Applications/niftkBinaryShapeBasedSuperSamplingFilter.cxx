/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <math.h>
#include <float.h>

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>
#include <itkCommandLineHelper.h>

#include <itkBinaryShapeBasedSuperSamplingFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>


struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_SWITCH, "iso", 0, "Create a volume with isotropic voxels."},

  {OPT_FLOAT, "sx", "xFactor", "The sampling factor in 'x' [1]."},
  {OPT_FLOAT, "sy", "yFactor", "The sampling factor in 'y' [1]."},
  {OPT_FLOAT, "sz", "zFactor", "The sampling factor in 'z' [1]."},

  {OPT_INT, "interp", "type", "The interpolator: "
   "1=Nearest, 2=Linear, 3=BSpline, 4=Sinc [2]."},

  {OPT_STRING, "o",    "filename", "The output sampled image."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to super-sample a mask in a specific dimension via shape based interpolation.\n"
  }
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_ISOTROPIC,

  O_SAMPLING_IN_X,
  O_SAMPLING_IN_Y,
  O_SAMPLING_IN_Z,

  O_INTERPOLATOR,

  O_OUTPUT_IMAGE,

  O_INPUT_IMAGE
};




// -------------------------------------------------------------------------
// arguments
// -------------------------------------------------------------------------

struct arguments
{
  bool flgDebug;
  bool flgVerbose;
  
  bool flgIsotropic;

  float sx;
  float sy;
  float sz;

  int interpolator;

  std::string fileInputImage;
  std::string fileOutputImage;

  arguments() {
    flgDebug = false;
    flgVerbose = false;

    flgIsotropic = false;

    interpolator = 2;

    sx = 1.;
    sy = 1.;
    sz = 1.;
  }
};


// -------------------------------------------------------------------------
// DoMain(arguments args)
// -------------------------------------------------------------------------

template <int Dimension>
int DoMain(arguments &args)
{
  double factor[3];


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  // Define the dimension of the images
  typedef float InputPixelType;
  typedef itk::Image<InputPixelType, Dimension> InputImageType;

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  typename FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName( args.fileInputImage );
  

  try
  { 
    std::cout << "Reading the input image" << std::endl;
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cerr << "ERROR: " << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Create the sampling filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef float OutputPixelType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;

  typedef itk::BinaryShapeBasedSuperSamplingFilter< InputImageType, OutputImageType > BinaryShapeBasedSuperSamplingFilterType;

  typename BinaryShapeBasedSuperSamplingFilterType::Pointer sampler = BinaryShapeBasedSuperSamplingFilterType::New();

  if ( args.flgVerbose )
  {
    sampler->VerboseOn();
  }

  if ( args.flgDebug )
  {
    sampler->DebugOn();
  }


  if ( args.flgIsotropic )
  {
    sampler->SetIsotropicVoxels( true );
  }
  else
  {
    factor[0] = args.sx;
    factor[1] = args.sy;
    factor[2] = args.sz;
      
    std::cout << "Sampling by: " 
              << factor[0] << ", " 
              << factor[1] << ", " 
              << factor[2] << std::endl;

    for ( unsigned int i=0; i<3; i++ )
    {
      if ( factor[i] > 1. )
      {
        std::cerr << "ERROR: Super-sampling factors must be less than or equal to one." << std::endl;
        return EXIT_FAILURE;
      }
      
      if ( factor[i] <= 0. )
      {
        std::cerr << "ERROR: Super-sampling factors greater than zero." << std::endl;
        return EXIT_FAILURE;
      }
    }

    sampler->SetSamplingFactors( factor );
  }

  switch ( args.interpolator )
  {
  case 1:
  {
    sampler->SetInterpolationType( itk::NEAREST );
    break;
  }
  case 2:
  {
    sampler->SetInterpolationType( itk::LINEAR );
    break;
  }
  case 3:
  {
    sampler->SetInterpolationType( itk::BSPLINE );
    break;
  }
  case 4:
  {
    sampler->SetInterpolationType( itk::SINC );
    break;
  }
  default:
  {
    std::cerr << "ERROR: Unrecognised interpolator" << std::endl;
    return EXIT_FAILURE;

    break;
  }
  }

  sampler->SetInput( imageReader->GetOutput() );
  
  try
  {
    std::cout << "Computing sampled image" << std::endl;
    sampler->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }


  // Write the sampled image?
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutputImage.length() != 0 ) {

    typedef itk::ImageFileWriter< OutputImageType > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( args.fileOutputImage );
    writer->SetInput( sampler->GetOutput() );

    try
      {
	std::cout << "Writing the output image." << std::endl;
	writer->Update();
      }
    catch (itk::ExceptionObject &e)
      {
	std::cerr << "ERROR: " << e << std::endl;
	return EXIT_FAILURE;
      }
  }
  
  return EXIT_SUCCESS;
}


// -------------------------------------------------------------------------
// main( int argc, char *argv[] )
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  int result;
 
  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  arguments args;

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_DEBUG,   args.flgDebug );
  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );

  CommandLineOptions.GetArgument( O_ISOTROPIC, args.flgIsotropic );

  CommandLineOptions.GetArgument( O_SAMPLING_IN_X, args.sx );
  CommandLineOptions.GetArgument( O_SAMPLING_IN_Y, args.sy );
  CommandLineOptions.GetArgument( O_SAMPLING_IN_Z, args.sz );

  CommandLineOptions.GetArgument( O_INTERPOLATOR, args.interpolator );

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, args.fileOutputImage );
  CommandLineOptions.GetArgument( O_INPUT_IMAGE,  args.fileInputImage );


  unsigned int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fileInputImage);

  if ( dims == 3 )
  {
    std::cout << "Image is 3D" << std::endl;
    result = DoMain<3>(args);
  }
  else
  {
    std::cerr << "ERROR: Unsupported image dimension" << std::endl;
    exit( EXIT_FAILURE );
  }

  return result;
}
