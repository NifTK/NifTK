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
#include <itkExtractImageFilter.h>

#include <niftkExtract2DSliceFrom3DImageCLP.h>

/*!
 * \file niftkExtract2DSliceFrom3DImage.cxx
 * \page niftkExtract2DSliceFrom3DImage
 * \section niftkExtract2DSliceFrom3DImageSummary Runs the ITK ExtractImageFilter.
 */
struct arguments
{
  std::string inputImage;
  std::string outputImage;
  std::vector<int> regionSize;
  std::vector<int> startingIndex;
};

template <int InputDimension, class PixelType>
int DoMain(arguments args)
{
  const   unsigned int OutputDimension = InputDimension - 1;

  typedef itk::Image< PixelType, InputDimension >                  InputImageType;
  typedef itk::Image< PixelType, OutputDimension >                 OutputImageType; 
  typedef itk::ImageFileReader< InputImageType >                   InputImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType >                  OutputImageWriterType;
  typedef itk::ExtractImageFilter<InputImageType, OutputImageType> ExtractImageFilterType;
  
  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);
  
  typename InputImageType::SizeType sizeType;
  typename InputImageType::IndexType indexType;
  typename InputImageType::RegionType regionType;
  
  for (unsigned int i = 0; i < InputDimension; i++)
    {
      sizeType[i] = args.regionSize[i];
      indexType[i] = args.startingIndex[i];
    }
  regionType.SetSize(sizeType);
  regionType.SetIndex(indexType);
  
  typename ExtractImageFilterType::Pointer filter = ExtractImageFilterType::New();
  filter->SetInput(imageReader->GetOutput());
  filter->SetExtractionRegion(regionType);
  
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
 * \brief Takes image and does shifting/scaling in ITK style.
 */
int main(int argc, char** argv)
{
  // To parse the command line args.
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;
  args.inputImage = inputImage.c_str();
  args.outputImage = outputImage.c_str();
  args.regionSize = regionSize;
  args.startingIndex = startingIndex;

  // Validate command line args
  if (inputImage.length() == 0 || outputImage.length() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  // The size of the arrays must be the same as the number if input dimensions.
  int dims = itk::PeekAtImageDimension(args.inputImage);
  if (dims != 3 || dims != 4)
  {
    std::cerr << "ERROR: Unsuported image dimension" << std::endl;
    return EXIT_FAILURE;
  }
  if (args.regionSize.size() != dims)
  {
    std::cerr << "ERROR: The size array must have length:" << dims << std::endl;
    return EXIT_FAILURE;
  }
  if (args.startingIndex.size() != dims)
  {
    std::cerr << "ERROR: The starting index array must have length:" << dims << std::endl;
    return EXIT_FAILURE;
  }

  // One dimension must be zero.
  bool foundZero = false;
  for (unsigned int i = 0; i < args.regionSize.size(); i++)
  {
    if (args.regionSize[i] == 0)
    {
      foundZero = true;
    }
  }
  if (!foundZero)
  {
    std::cerr << "ERROR: At least one size must be zero!" << std::endl;
    return EXIT_FAILURE;
  }

  if (dims == 3)
  {
    switch (itk::PeekAtComponentType(args.inputImage))
      {
      case itk::ImageIOBase::UCHAR:
        DoMain<3, unsigned char>(args);
        break;
      case itk::ImageIOBase::CHAR:
        DoMain<3, char>(args);
        break;
      case itk::ImageIOBase::USHORT:
        DoMain<3, unsigned short>(args);
        break;
      case itk::ImageIOBase::SHORT:
        DoMain<3, short>(args);
        break;
      case itk::ImageIOBase::UINT:
        DoMain<3, unsigned int>(args);
        break;
      case itk::ImageIOBase::INT:
        DoMain<3, int>(args);
        break;
      case itk::ImageIOBase::ULONG:
        DoMain<3, unsigned long>(args);
        break;
      case itk::ImageIOBase::LONG:
        DoMain<3, long>(args);
        break;
      case itk::ImageIOBase::FLOAT:
        DoMain<3, float>(args);
        break;
      case itk::ImageIOBase::DOUBLE:
        DoMain<3, double>(args);
        break;
      default:
        std::cerr << "non standard pixel format" << std::endl;
        return EXIT_FAILURE;
      }
  }
  else if (dims == 4)
  {
    switch (itk::PeekAtComponentType(args.inputImage))
      {
      case itk::ImageIOBase::UCHAR:
        DoMain<4, unsigned char>(args);
        break;
      case itk::ImageIOBase::CHAR:
        DoMain<4, char>(args);
        break;
      case itk::ImageIOBase::USHORT:
        DoMain<4, unsigned short>(args);
        break;
      case itk::ImageIOBase::SHORT:
        DoMain<4, short>(args);
        break;
      case itk::ImageIOBase::UINT:
        DoMain<4, unsigned int>(args);
        break;
      case itk::ImageIOBase::INT:
        DoMain<4, int>(args);
        break;
      case itk::ImageIOBase::ULONG:
        DoMain<4, unsigned long>(args);
        break;
      case itk::ImageIOBase::LONG:
        DoMain<4, long>(args);
        break;
      case itk::ImageIOBase::FLOAT:
        DoMain<4, float>(args);
        break;
      case itk::ImageIOBase::DOUBLE:
        DoMain<4, double>(args);
        break;
      default:
        std::cerr << "non standard pixel format" << std::endl;
        return EXIT_FAILURE;
      }
  }

  return EXIT_SUCCESS; 
}
