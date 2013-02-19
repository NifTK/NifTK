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
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBinaryCrossStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"

/*!
 * \file niftkErode.cxx
 * \page niftkErode
 * \section niftkErodeSummary Runs the ITK BinaryErodeImageFilter, using a BinaryCrossStructuringElement.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs the ITK BinaryErodeImageFilter, using a BinaryCrossStructuringElement." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input mask image " << std::endl;
    std::cout << "    -o    <filename>        Output mask image" << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -r    <int>   [1]       Radius of structuring element" << std::endl;
    std::cout << "    -it   <int>   [1]       Iterations" << std::endl;
    std::cout << "    -e    <int>   [1]       Eroded value" << std::endl;
    std::cout << "    -b    <int>   [0]       Background value" << std::endl;
  }

/**
 * \brief Takes image and uses ITK to do dilation.
 */
int main(int argc, char** argv)
{
  const   unsigned int Dimension = 3;
  typedef short        PixelType;

  // Define command line params
  std::string inputImage;
  std::string outputImage;
  int radius = 1;
  int iterations = 1;
  int erodeValue = 1;
  int backgroundValue = 0;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      inputImage=argv[++i];
      std::cout << "Set -i=" << inputImage << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      outputImage=argv[++i];
      std::cout << "Set -o=" << outputImage << std::endl;
    }
    else if(strcmp(argv[i], "-it") == 0){
      iterations=atoi(argv[++i]);
      std::cout << "Set -it=" << niftk::ConvertToString(iterations) << std::endl;
    }
    else if(strcmp(argv[i], "-e") == 0){
      erodeValue=atoi(argv[++i]);
      std::cout << "Set -e=" << niftk::ConvertToString(erodeValue) << std::endl;
    }
    else if(strcmp(argv[i], "-b") == 0){
      backgroundValue=atoi(argv[++i]);
      std::cout << "Set -b=" << niftk::ConvertToString(backgroundValue) << std::endl;
    }
    else if(strcmp(argv[i], "-r") == 0){
      backgroundValue=atoi(argv[++i]);
      std::cout << "Set -r=" << niftk::ConvertToString(backgroundValue) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }
  }

  // Validate command line args
  if (inputImage.length() == 0 || outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  typedef itk::Image< PixelType, Dimension >     InputImageType;
  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::ImageFileWriter< InputImageType > OutputImageWriterType;

  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

  imageReader->SetFileName(inputImage);

  typedef itk::BinaryCrossStructuringElement<InputImageType::PixelType, InputImageType::ImageDimension> StructuringElementType;
  StructuringElementType element;
  element.SetRadius(radius);
  element.CreateStructuringElement();

  typedef itk::BinaryErodeImageFilter<InputImageType, InputImageType, StructuringElementType> ErodeImageFilterType;
  ErodeImageFilterType::Pointer filter = ErodeImageFilterType::New();
  filter->SetInput(imageReader->GetOutput());
  filter->SetKernel(element);
  filter->SetErodeValue(erodeValue);
  filter->SetBackgroundValue(backgroundValue);
  filter->SetBoundaryToForeground(false);

  std::cout << "Filtering with radius:" << niftk::ConvertToString(radius)
    << ", iterations:" << niftk::ConvertToString(iterations)
    << ", erodeValue:" << niftk::ConvertToString(erodeValue)
    << ", backgroundValue:" << niftk::ConvertToString(backgroundValue) << std::endl;

  try
  {
    if (iterations > 1)
      {
        for (int i = 0; i < iterations - 1; i++)
          {
            filter->Update();
            InputImageType::Pointer image = filter->GetOutput();
            image->DisconnectPipeline();
            filter->SetInput(image);
          }
        filter->Update();
      }

    imageWriter->SetFileName(outputImage);
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
