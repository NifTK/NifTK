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
#include <itkMultiplyImageFilter.h>

#include <niftkMultiplyCLP.h>

/*!
 * \file niftkMultiply.cxx
 * \page niftkMultiply
 * \section niftkMultiplySummary Multiplies image 1 by image 2 on a voxel-by-voxel basis, without checking image sizes.
 */
struct arguments
{
  std::string inputImage1;
  std::string inputImage2;
  std::string outputImage;    
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  typedef typename itk::MultiplyImageFilter<InputImageType, InputImageType> MultiplyFilterType;
  
  typename InputImageReaderType::Pointer input1Reader = InputImageReaderType::New();
  typename InputImageReaderType::Pointer input2Reader = InputImageReaderType::New();
  typename MultiplyFilterType::Pointer filter = MultiplyFilterType::New();
  
  input1Reader->SetFileName(args.inputImage1);
  input2Reader->SetFileName(args.inputImage2);
  
  filter->SetInput1(input1Reader->GetOutput());
  filter->SetInput2(input2Reader->GetOutput());
  
  typename MultiplyFilterType::Pointer subtractFilter = MultiplyFilterType::New();  
  subtractFilter->SetInput1(input1Reader->GetOutput());
  subtractFilter->SetInput2(input2Reader->GetOutput());
  
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
 * \brief Takes image1 and image2 and multiplies them together.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;
  args.inputImage1=inputImage1.c_str();
  args.inputImage2=inputImage2.c_str();
  args.outputImage=outputImage.c_str();

  // Validate command line args
  if (args.inputImage1.length() == 0 ||
      args.inputImage2.length() == 0 ||
      args.outputImage.length() == 0)
    {
      return EXIT_FAILURE;
    }

  int dims = itk::PeekAtImageDimension(args.inputImage1);
  if (dims != 2 && dims != 3)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }
  
  int result;

  switch (itk::PeekAtComponentType(args.inputImage1))
    {
    case itk::ImageIOBase::UCHAR:
      if (dims == 2)
        {
          result = DoMain<2, unsigned char>(args);  
        }
      else
        {
          result = DoMain<3, unsigned char>(args);
        }
      break;
    case itk::ImageIOBase::CHAR:
      if (dims == 2)
        {
          result = DoMain<2, char>(args);  
        }
      else
        {
          result = DoMain<3, char>(args);
        }
      break;
    case itk::ImageIOBase::USHORT:
      if (dims == 2)
        {
          result = DoMain<2, unsigned short>(args);  
        }
      else
        {
          result = DoMain<3, unsigned short>(args);
        }
      break;
    case itk::ImageIOBase::SHORT:
      if (dims == 2)
        {
          result = DoMain<2, short>(args);  
        }
      else
        {
          result = DoMain<3, short>(args);
        }
      break;
    case itk::ImageIOBase::UINT:
      if (dims == 2)
        {
          result = DoMain<2, unsigned int>(args);  
        }
      else
        {
          result = DoMain<3, unsigned int>(args);
        }
      break;
    case itk::ImageIOBase::INT:
      if (dims == 2)
        {
          result = DoMain<2, int>(args);  
        }
      else
        {
          result = DoMain<3, int>(args);
        }
      break;
    case itk::ImageIOBase::ULONG:
      if (dims == 2)
        {
          result = DoMain<2, unsigned long>(args);  
        }
      else
        {
          result = DoMain<3, unsigned long>(args);
        }
      break;
    case itk::ImageIOBase::LONG:
      if (dims == 2)
        {
          result = DoMain<2, long>(args);  
        }
      else
        {
          result = DoMain<3, long>(args);
        }
      break;
    case itk::ImageIOBase::FLOAT:
      if (dims == 2)
        {
          result = DoMain<2, float>(args);  
        }
      else
        {
          result = DoMain<3, float>(args);
        }
      break;
    case itk::ImageIOBase::DOUBLE:
      if (dims == 2)
        {
          result = DoMain<2, double>(args);  
        }
      else
        {
          result = DoMain<3, double>(args);
        }
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
}
