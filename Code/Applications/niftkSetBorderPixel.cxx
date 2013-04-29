/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

/*!
 * \file niftkSetBorderPixel.cxx
 * \page niftkSetBorderPixel
 * \section niftkSetBorderPixelSummary Simply sets the border pixel to the specified value.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Simply sets the border pixel to the specified value." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -v    <float> [255]     Border value" << std::endl;
  }

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  float value;
};

template <int Dimension, class PixelType>
int DoMain(arguments args)
{
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;

  try
  {
    typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
    imageReader->SetFileName(args.inputImage);
    imageReader->Update();

    typename InputImageType::Pointer outputImage = InputImageType::New();
    outputImage->SetRegions(imageReader->GetOutput()->GetLargestPossibleRegion());
    outputImage->SetOrigin(imageReader->GetOutput()->GetOrigin());
    outputImage->SetSpacing(imageReader->GetOutput()->GetSpacing());
    outputImage->SetDirection(imageReader->GetOutput()->GetDirection());
    outputImage->Allocate();
    outputImage->FillBuffer(0);

    itk::ImageRegionConstIterator<InputImageType> inputIterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionIterator<InputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());

    typename InputImageType::IndexType imageIndex;
    typename InputImageType::SizeType imageSize;

    imageSize = outputImage->GetLargestPossibleRegion().GetSize();

    unsigned int i = 0;
    bool isBorder = false;

    for (inputIterator.GoToBegin(), outputIterator.GoToBegin();
         !inputIterator.IsAtEnd();
         ++inputIterator, ++outputIterator)
    {
      imageIndex = inputIterator.GetIndex();

      isBorder = false;
      for (i = 0; i < Dimension; i++)
      {
        if ((long int)imageIndex[i] == (long int)0 || (long int)imageIndex[i] == (long int)(imageSize[i]-1))
        {
          isBorder = true;
        }
      }
      if (isBorder)
      {
        outputIterator.Set((PixelType)args.value);
      }
      else
      {
        outputIterator.Set(inputIterator.Get());
      }
    }

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
    imageWriter->SetFileName(args.outputImage);
    imageWriter->SetInput(outputImage);
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
 * \brief Takes image1 and image2 and adds them together
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.value = 255;

  

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
    else if(strcmp(argv[i], "-v") == 0){
      args.value=atof(argv[++i]);
      std::cout << "Set -v=" << niftk::ConvertToString(args.value) << std::endl;
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
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
}
