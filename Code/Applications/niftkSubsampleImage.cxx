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

#include <niftkSubsampleImageCLP.h>


// -------------------------------------------------------------------------
// arguments
// -------------------------------------------------------------------------

struct arguments
{
  float subx;
  float suby;
  float subz;

  std::string fileInputImage;
  std::string fileOutputImage;

  arguments() {

    subx = 1.;
    suby = 1.;
    subz = 1.;
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
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Create the subsampling filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef float OutputPixelType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;

  typedef itk::SubsampleImageFilter< InputImageType, OutputImageType > SubsampleImageFilterType;

  typename SubsampleImageFilterType::Pointer sampler = SubsampleImageFilterType::New();

  factor[0] = args.subx;
  factor[1] = args.suby;
  factor[2] = args.subz;

  std::cout << "Subsampling by: " 
            << factor[0] << ", " 
            << factor[1] << ", " 
            << factor[2] << std::endl;

  sampler->SetSubsamplingFactors( factor );

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


// -------------------------------------------------------------------------
// main( int argc, char *argv[] )
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  // To pass around command line args
  arguments args;

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( fileInputImage.length() == 0 || fileOutputImage.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  args.subx = subx;
  args.suby = suby;
  args.subz = subz;

  args.fileInputImage  = fileInputImage;
  args.fileOutputImage = fileOutputImage;
  

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
