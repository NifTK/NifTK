/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNegateImageFilter.h>
#include <itkCastImageFilter.h>

#include <niftkNegateImageCLP.h>

/*!
 * \file niftkNegateImage.cxx
 * \page niftkNegateImage
 * \section niftkNegateImageSummary Runs filter NegateImageFilter on an image by computing Inew = Imax - I + Imin
 *
 * This program uses ITK ImageFileReader to load an image, and then uses NegateImageFilter to negate its intensities before writing the output with ITK ImageFileWriter.
 *
 * \li Dimensions: 2,3.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkNegateImageCaveats Caveats
 * \li None
 */
typedef struct arguments
{
  std::string inputImage;
  std::string outputImage;  
} Arguments;

template <int Dimension, class InputPixelType, class OutputPixelType> 
int DoMain(Arguments args)
{  
  typedef typename itk::Image< InputPixelType,  Dimension > InputImageType;   
  typedef typename itk::Image< OutputPixelType, Dimension > OutputImageType;   

  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::CastImageFilter< InputImageType, OutputImageType > CastFilterType;
  typedef typename itk::NegateImageFilter<OutputImageType, OutputImageType> NegateFilterType;
  typedef typename itk::ImageFileWriter< OutputImageType > OutputImageWriterType;

  std::cout << "Input image:  " << args.inputImage << std::endl
            << "Output image: " << args.outputImage << std::endl;

  try
  {

    typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
    imageReader->SetFileName(args.inputImage);
    imageReader->Update(); 

    typename CastFilterType::Pointer caster = CastFilterType::New();
    caster->SetInput(imageReader->GetOutput());
    caster->Update(); 

    typename NegateFilterType::Pointer filter = NegateFilterType::New();
    filter->SetInput(caster->GetOutput());
    filter->Update(); 

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
    imageWriter->SetFileName(args.outputImage);
    imageWriter->SetInput(caster->GetOutput());
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
 * \brief Takes the input image and negates it using NegateImageFilter
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // To pass around command line args
  Arguments args;
  args.inputImage  = inputImage.c_str();
  args.outputImage = outputImage.c_str();

  // Validate command line args
  if (args.inputImage.length() == 0 ||
      args.outputImage.length() == 0)
    {
      return EXIT_FAILURE;
    }

  int dims = itk::PeekAtImageDimension(args.inputImage);
  if (dims != 2 && dims != 3)
    {
      std::cout << "Unsupported image dimension" << std::endl;
      return EXIT_FAILURE;
    }
  
  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
    {
    case itk::ImageIOBase::UCHAR:
      if (dims == 2)
        {
          result = DoMain<2, unsigned char, short>(args);  
        }
      else
        {
          result = DoMain<3, unsigned char, short>(args);
        }
      break;
    case itk::ImageIOBase::CHAR:
      if (dims == 2)
        {
          result = DoMain<2, char, char>(args);  
        }
      else
        {
          result = DoMain<3, char, char>(args);
        }
      break;
    case itk::ImageIOBase::USHORT:
      if (dims == 2)
        {
          std::cout << "Converting unsigned short to long" << std::endl;
          result = DoMain<2, unsigned short, long>(args);  
        }
      else
        {
          result = DoMain<3, unsigned short, long>(args);
        }
      break;
    case itk::ImageIOBase::SHORT:
      if (dims == 2)
        {
          result = DoMain<2, short, short>(args);  
        }
      else
        {
          result = DoMain<3, short, short>(args);
        }
      break;
    case itk::ImageIOBase::UINT:
      if (dims == 2)
        {
          result = DoMain<2, unsigned long, float>(args);  
        }
      else
        {
          result = DoMain<3, unsigned long, float>(args);
        }
      break;
    case itk::ImageIOBase::INT:
      if (dims == 2)
        {
          result = DoMain<2, long, long>(args);  
        }
      else
        {
          result = DoMain<3, long, long>(args);
        }
      break;
    case itk::ImageIOBase::ULONG:
      if (dims == 2)
        {
          result = DoMain<2, unsigned long, float>(args);  
        }
      else
        {
          result = DoMain<3, unsigned long, float>(args);
        }
      break;
    case itk::ImageIOBase::LONG:
      if (dims == 2)
        {
          result = DoMain<2, long, long>(args);  
        }
      else
        {
          result = DoMain<3, long, long>(args);
        }
      break;
    case itk::ImageIOBase::FLOAT:
      if (dims == 2)
        {
          result = DoMain<2, float, float>(args);  
        }
      else
        {
          result = DoMain<3, float, float>(args);
        }
      break;
    case itk::ImageIOBase::DOUBLE:
      if (dims == 2)
        {
          result = DoMain<2, double, double>(args);  
        }
      else
        {
          result = DoMain<3, double, double>(args);
        }
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
}
