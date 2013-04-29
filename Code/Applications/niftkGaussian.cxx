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
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDiscreteGaussianImageFilter.h"

/*!
 * \file niftkGaussian.cxx
 * \page niftkGaussian
 * \section niftkGaussianSummary Runs the ITK DiscreteGaussianImageFilter.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs the ITK DiscreteGaussianImageFilter." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;   
    std::cout << "    -v    <float>  [0]      Variance in millimetres" << std::endl;
    std::cout << "    -w    <int>    [5]      Maximum kernel width in voxels" << std::endl;
  }


/**
 * \brief Takes image and uses ITK to do gaussian blurring.
 */
int main(int argc, char** argv)
{
  
  const   unsigned int Dimension = 3;
  typedef float        PixelType;

  // Define command line params
  std::string inputImage;
  std::string outputImage;
  double variance=0;
  int width=5;
  

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
    else if(strcmp(argv[i], "-v") == 0){
      variance=atof(argv[++i]);
      std::cout << "Set -v=" << niftk::ConvertToString(variance) << std::endl;
    }
    else if(strcmp(argv[i], "-w") == 0){
      width=atoi(argv[++i]);
      std::cout << "Set -w=" << niftk::ConvertToString(width) << std::endl;
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
  
  if(variance < 0 ){
    std::cerr << argv[0] << "\tThe variance must be >= 0" << std::endl;
    return -1;
  }

  if(width <= 0 ){
    std::cerr << argv[0] << "\tThe width must be > 0" << std::endl;
    return -1;
  }

  typedef itk::Image< PixelType, Dimension >     InputImageType;   
  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  typedef itk::DiscreteGaussianImageFilter<InputImageType, InputImageType> GaussianFilterType;
  
  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  GaussianFilterType::Pointer filter = GaussianFilterType::New();
  
  imageReader->SetFileName(inputImage);
  
  filter->SetInput(imageReader->GetOutput());
  filter->SetVariance(variance);
  filter->SetMaximumKernelWidth(width);
  
  std::cout << "Filtering with variance:" << niftk::ConvertToString(variance)
      << ", Width:" << niftk::ConvertToString(width) << std::endl;
  
  imageWriter->SetFileName(outputImage);
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
