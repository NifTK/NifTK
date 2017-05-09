/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNifTKImageIOFactory.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkCommandLineHelper.h>

/*!
 * \file niftkThreshold.cxx
 * \page niftkThreshold
 * \section niftkThresholdSummary Runs the ITK BinaryThresholdImageFilter.
 */

struct niftk::CommandLineArgumentDescription clArgList[] = 
{
  {OPT_STRING|OPT_REQ, "i", "filename", "Input image."},
  {OPT_STRING|OPT_REQ, "o", "filename", "Output image."},

  {OPT_FLOAT, "u", "float", "Upper value."},
  {OPT_FLOAT, "l", "float", "Lower value."},
  {OPT_FLOAT, "in", "float", "[1] Inside value."},
  {OPT_FLOAT, "out", "float", "[0] Outside value."},

  {OPT_DONE, NULL, NULL, 
   "Runs the ITK BinaryThresholdImageFilter on a 2D or 3D image. \n"
  }
};

enum
{
  O_INPUT_IMAGE, 

  O_OUTPUT_IMAGE, 

  O_UPPER_VALUE, 

  O_LOWER_VALUE,

  O_FOREGROUND_VALUE,

  O_BACKGROUND_VALUE
};

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  float upper;
  float lower;
  float inside;
  float outside;
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  

  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  typedef typename itk::BinaryThresholdImageFilter<InputImageType, InputImageType> BinaryThresholdFilterType;
  
  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);

  PixelType min = 0;
  PixelType max = 0;
  
  if (args.upper == std::numeric_limits<float>::max())
    {
      max = std::numeric_limits<PixelType>::max();
    }
  else
    {
      max = (PixelType)args.upper;
    }
  if (args.lower == std::numeric_limits<float>::min())
    {
      min = std::numeric_limits<PixelType>::min();
    }
  else
    {
      min = (PixelType)args.lower;
    }

  typename BinaryThresholdFilterType::Pointer filter = BinaryThresholdFilterType::New();
  filter->SetInput(imageReader->GetOutput());
  filter->SetOutsideValue((PixelType)args.outside);
  filter->SetInsideValue((PixelType)args.inside);
  filter->SetUpperThreshold(max);
  filter->SetLowerThreshold(min);
  
  
  std::cout << "Filtering with upper:" << niftk::ConvertToString((double)max)
    << ", lower:" << niftk::ConvertToString((double)min)
    << ", inside:" << niftk::ConvertToString(args.inside)
    << ", outside:" << niftk::ConvertToString(args.outside) << std::endl;
  
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
 * \brief Takes image and does binary thresholding in ITK style.
 */
int main(int argc, char** argv)
{
  itk::NifTKImageIOFactory::Initialize();

  // To pass around command line args
  struct arguments args;
  
  // Define defaults
  args.upper = std::numeric_limits<float>::max();
  args.lower = std::numeric_limits<float>::min();
  args.inside = 1;
  args.outside = 0;
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, false);

  CommandLineOptions.GetArgument( O_INPUT_IMAGE, args.inputImage );

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, args.outputImage );

  CommandLineOptions.GetArgument( O_UPPER_VALUE, args.upper );
  
  CommandLineOptions.GetArgument( O_LOWER_VALUE, args.lower );

  CommandLineOptions.GetArgument( O_FOREGROUND_VALUE, args.inside );
  
  CommandLineOptions.GetArgument( O_BACKGROUND_VALUE, args.outside );
  
  int dims = itk::PeekAtImageDimension(args.inputImage);
  if (dims != 2 && dims != 3)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }
  
  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
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
