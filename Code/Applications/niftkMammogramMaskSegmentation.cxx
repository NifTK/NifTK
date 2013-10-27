/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <itkLogHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMammogramMaskSegmentationImageFilter.h>

#include <niftkMammogramMaskSegmentationCLP.h>

/*!
 * \file niftkMammogramMaskSegmentation.cxx
 * \page niftkMammogramMaskSegmentation
 * \section niftkMammogramMaskSegmentationSummary Segments a mammogram generating a binary mask corresponding to the breast area.
 *
 * This program uses ITK ImageFileReader to load an image, and then uses MammogramMaskSegmentationImageFilter to segment the breast reagion before writing the output with ITK ImageFileWriter.
 *
 * \li Dimensions: 2.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkMammogramMaskSegmentationCaveats Caveats
 * \li None
 */
struct arguments
{
  std::string inputImage;
  std::string outputImage;  
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  typedef typename itk::MammogramMaskSegmentationImageFilter<InputImageType> MammogramMaskSegmentationImageFilterType;

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);

  typename MammogramMaskSegmentationImageFilterType::Pointer filter = MammogramMaskSegmentationImageFilterType::New();
  filter->SetInput(imageReader->GetOutput());
  
  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput(filter->GetOutput());
  
  try
  {
    imageWriter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  return EXIT_SUCCESS;
}

/**
 * \brief Takes the input image and inverts it using InvertIntensityBetweenMaxAndMinImageFilter
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;
  args.inputImage=inputImage.c_str();
  args.outputImage=outputImage.c_str();

  std::cout << "Input image:  " << args.inputImage << std::endl
            << "Output image: " << args.outputImage << std::endl;

  // Validate command line args
  if (args.inputImage.length() == 0 ||
      args.outputImage.length() == 0)
  {
    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.inputImage);
  if (dims != 2)
  {
    std::cout << "ERROR: Unsupported image dimension" << std::endl;
    return EXIT_FAILURE;
  }
  else if (dims == 2)
  {
    std::cout << "Input is 2D" << std::endl;
  }
   
  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
  {
  case itk::ImageIOBase::UCHAR:
    std::cout << "Input is UNSIGNED CHAR" << std::endl;
    result = DoMain<2, unsigned char>(args);  
    break;

  case itk::ImageIOBase::CHAR:
    result = DoMain<2, char>(args);  
    break;

  case itk::ImageIOBase::USHORT:
    std::cout << "Input is UNSIGNED SHORT" << std::endl;
    result = DoMain<2, unsigned short>(args);  
    break;

  case itk::ImageIOBase::SHORT:
    std::cout << "Input is SHORT" << std::endl;
    result = DoMain<2, short>(args);  
    break;

  case itk::ImageIOBase::UINT:
    std::cout << "Input is UNSIGNED INT" << std::endl;
    result = DoMain<2, unsigned int>(args);  
    break;

  case itk::ImageIOBase::INT:
    std::cout << "Input is SIGNED INT" << std::endl;
    result = DoMain<2, int>(args);  
    break;

  case itk::ImageIOBase::ULONG:
    std::cout << "Input is UNSIGNED LONG" << std::endl;
    result = DoMain<2, unsigned long>(args);  
    break;

  case itk::ImageIOBase::LONG:
    std::cout << "Input is LONG" << std::endl;
    result = DoMain<2, long>(args);  
    break;

  case itk::ImageIOBase::FLOAT:
    std::cout << "Input is FLOAT" << std::endl;
    result = DoMain<2, float>(args);  
    result = DoMain<3, float>(args);
    break;

  case itk::ImageIOBase::DOUBLE:
    std::cout << "Input is DOUBLE" << std::endl;
    result = DoMain<2, double>(args);  
    break;

  default:
    std::cerr << "ERROR: non standard pixel format" << std::endl;
    return EXIT_FAILURE;
  }
  return result;
}
