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
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkCommandLineHelper.h>

/*!
 * \file niftkThreshold.cxx
 * \page niftkThreshold
 * \section niftkThresholdSummary Runs the ITK BinaryThresholdImageFilter.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs the ITK BinaryThresholdImageFilter." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>            Input image " << std::endl;
    std::cout << "    -o    <filename>            Output image" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl; 
    std::cout << "    -u    <int> [data type max] Highest value" << std::endl;
    std::cout << "    -l    <int> [data type min] Lowest value" << std::endl;
    std::cout << "    -in   <int> [1]             Inside value" << std::endl;
    std::cout << "    -out  <int> [0]             Outside value" << std::endl;
  }

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
  // To pass around command line args
  struct arguments args;
  
  // Define defaults
  args.upper = std::numeric_limits<float>::max();
  args.lower = std::numeric_limits<float>::min();
  args.inside = 1;
  args.outside = 0;
  
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputImage=argv[++i];
      std::cout << "Set -i=" << args.inputImage << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputImage=argv[++i];
      std::cout << "Set -o=" << args.outputImage << std::endl;
    }
    else if(strcmp(argv[i], "-u") == 0){
      args.upper=atof(argv[++i]);
      std::cout << "Set -u=" << niftk::ConvertToString(args.upper) << std::endl;
    }
    else if(strcmp(argv[i], "-l") == 0){
      args.lower=atof(argv[++i]);
      std::cout << "Set -l=" << niftk::ConvertToString(args.lower) << std::endl;
    }
    else if(strcmp(argv[i], "-in") == 0){
      args.inside=atof(argv[++i]);
      std::cout << "Set -in=" << niftk::ConvertToString(args.inside) << std::endl;
    }
    else if(strcmp(argv[i], "-out") == 0){
      args.outside=atof(argv[++i]);
      std::cout << "Set -out=" << niftk::ConvertToString(args.outside) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

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
