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
    imageWriter->SetInput(filter->GetOutput());
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
  args.inputImage  = inputImage;
  args.outputImage = outputImage;

  std::cout << "Input image:  " << args.inputImage << std::endl
            << "Output image: " << args.outputImage << std::endl;

  // Validate command line args
  if (args.inputImage.length() == 0 ||
      args.outputImage.length() == 0)
  {
    std::cerr << "ERROR: Input and output images must be set" << std::endl;
    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.inputImage);
  if ( (dims != 2) && (dims != 3) )
  {
    std::cout << "ERROR: Unsupported image dimension" << std::endl;
    return EXIT_FAILURE;
  }
  else if (dims == 2)
  {
    std::cout << "Input is 2D" << std::endl;
  }
  else
  {
    std::cout << "Input is 3D" << std::endl;
  }
  
  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
  {
  case itk::ImageIOBase::UCHAR:
  {
    std::cout << "Converting UNSIGNED CHAR to SHORT INT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, unsigned char, short int>(args);  
    }
    else
    {
      result = DoMain<3, unsigned char, short int>(args);
    }
    break;
  }
  case itk::ImageIOBase::CHAR:
  {
    std::cout << "Input is CHAR" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, char, char>(args);  
    }
    else
    {
      result = DoMain<3, char, char>(args);
    }
    break;
  }
  case itk::ImageIOBase::USHORT:
  {
    std::cout << "Converting UNSIGNED SHORT INT to INT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, unsigned short int, int>(args);  
    }
    else
    {
      result = DoMain<3, unsigned short int, int>(args);
    }
    break;
  }
  case itk::ImageIOBase::SHORT:
  {
    std::cout << "Input is SHORT INT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, short int, short int>(args);  
    }
    else
    {
      result = DoMain<3, short int, short int>(args);
    }
    break;
  }
  case itk::ImageIOBase::UINT:
  {
    std::cout << "Converting UNSIGNED INT to FLOAT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, unsigned int, float>(args);  
    }
    else
    {
      result = DoMain<3, unsigned int, float>(args);
    }
    break;
  }
  case itk::ImageIOBase::INT:
  {
    std::cout << "Input is INT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, int, int>(args);  
    }
    else
    {
      result = DoMain<3, int, int>(args);
    }
    break;
  }
  case itk::ImageIOBase::ULONG:
  {
    std::cout << "Converting UNSIGNED LONG to FLOAT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, unsigned int, float>(args);  
    }
    else
    {
      result = DoMain<3, unsigned int, float>(args);
    }
    break;
  }
  case itk::ImageIOBase::LONG:
  {
    std::cout << "Input is LONG" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, int, int>(args);  
    }
    else
    {
      result = DoMain<3, int, int>(args);
    }
    break;
  }
  case itk::ImageIOBase::FLOAT:
  {
    std::cout << "Input is FLOAT" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, float, float>(args);  
    }
    else
    {
      result = DoMain<3, float, float>(args);
    }
    break;
  }
  case itk::ImageIOBase::DOUBLE:
  {
    std::cout << "Input is DOUBLE" << std::endl;
    if (dims == 2)
    {
      result = DoMain<2, double, double>(args);  
    }
    else
    {
      result = DoMain<3, double, double>(args);
    }
    break;
  }
  default:
  {
    std::cerr << "ERROR: Non standard pixel format" << std::endl;
    return EXIT_FAILURE;
  }
  }
  return result;
}
