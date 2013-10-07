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

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImage.h>

#include <itkOtsuThresholdImageFilter.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>

#include <niftkOtsuThresholdImageCLP.h>


// -------------------------------------------------------------------------
// arguments
// -------------------------------------------------------------------------

struct arguments
{
  std::string fileInputImage;
  std::string fileOutputMask;
};


// -------------------------------------------------------------------------
// DoMain(arguments args)
// -------------------------------------------------------------------------

template <int Dimension, class OutputPixelType>
int DoMain(arguments &args)
{

  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

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


  // Create a mask by thresholding
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef unsigned char MaskPixelType;
  typedef itk::Image<MaskPixelType, Dimension> MaskImageType;
  typedef itk::OtsuThresholdImageFilter< InputImageType, 
                                         MaskImageType > OtsuThresholdImageFilterType;

  typename OtsuThresholdImageFilterType::Pointer thresholder = OtsuThresholdImageFilterType::New();

  thresholder->SetInput( imageReader->GetOutput() );
  
  try
  {
    std::cout << "Thresholding to obtain image mask" << std::endl;
    thresholder->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }


  // Invert the mask
  // ~~~~~~~~~~~~~~~

  typedef typename itk::InvertIntensityBetweenMaxAndMinImageFilter<MaskImageType> InvertFilterType;

  typename InvertFilterType::Pointer inverter = InvertFilterType::New();

  inverter->SetInput( thresholder->GetOutput() );
  
  try
  {
    std::cout << "Inverting the mask" << std::endl;
    inverter->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }


  // Write the mask image to a file?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutputMask.length() != 0 ) 
  {
    typedef itk::Image<MaskPixelType, Dimension> OutputMaskType;

    typedef itk::ImageFileWriter< OutputMaskType > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( args.fileOutputMask );
    writer->SetInput( inverter->GetOutput() );

    try
    {
      std::cout << "Writing the mask to image: " << args.fileOutputMask << std::endl;
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

  if ( fileInputImage.length() == 0 || fileOutputMask.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  args.fileInputImage  = fileInputImage;
  args.fileOutputMask = fileOutputMask;
  

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fileInputImage);
  if (dims != 3)
  {
    std::cout << "ERROR: Unsupported image dimension, image must be 3D" << std::endl;
    return EXIT_FAILURE;
  }
  
  int result;

  switch (itk::PeekAtComponentType(args.fileInputImage))
  {
  case itk::ImageIOBase::UCHAR:
    result = DoMain<3, unsigned char>(args);
    break;

  case itk::ImageIOBase::CHAR:
    result = DoMain<3, char>(args);
    break;

  case itk::ImageIOBase::USHORT:
    result = DoMain<3, unsigned short>(args);
    break;

  case itk::ImageIOBase::SHORT:
    result = DoMain<3, short>(args);
    break;

  case itk::ImageIOBase::UINT:
    result = DoMain<3, unsigned int>(args);
    break;

  case itk::ImageIOBase::INT:
    result = DoMain<3, int>(args);
    break;

  case itk::ImageIOBase::ULONG:
    result = DoMain<3, unsigned long>(args);
    break;

  case itk::ImageIOBase::LONG:
    result = DoMain<3, long>(args);
    break;

  case itk::ImageIOBase::FLOAT:
    result = DoMain<3, float>(args);
    break;

  case itk::ImageIOBase::DOUBLE:
    result = DoMain<3, double>(args);
    break;

  default:
    std::cerr << "ERROR: Unsupported pixel format" << std::endl;
    return EXIT_FAILURE;
  }
  return result;
}
