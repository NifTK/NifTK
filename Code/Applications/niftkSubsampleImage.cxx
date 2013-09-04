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

#include <itkSubsampleImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImage.h>


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_DOUBLEx3|OPT_REQ, "sub", "value", "The subsampling factor (greater than 1) eg. 1.5,1.5,2."},

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

  double *factor;

  std::string fileOutputImage;
  std::string fileInputImage;
};


template <int Dimension>
int DoMain(arguments args)
{
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
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Create the subsampling filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef float OutputPixelType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;

  typedef itk::SubsampleImageFilter< InputImageType, OutputImageType > SubsampleImageFilterType;

  typename SubsampleImageFilterType::Pointer sampler = SubsampleImageFilterType::New();

  sampler->SetSubsamplingFactors( args.factor );

  sampler->SetInput( imageReader->GetOutput() );
  
  try
  {
    std::cout << "Computing subsampled image" << std::endl;
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
	std::cout << "Writing the output image." << std::endl;
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
  args.factor = 0;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );

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
        std::cout << "Images are 2D" << std::endl;
        result = DoMain<2>(args);
        break;
      case 3:
        std::cout << "Images are 3D" << std::endl;
        result = DoMain<3>(args);
      break;
      default:
        std::cout << "Unsuported image dimension" << std::endl;
        exit( EXIT_FAILURE );
    }
  return result;
}
