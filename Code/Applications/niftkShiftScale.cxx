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
#include <itkShiftScaleImageFilter.h>

/*!
 * \file niftkShiftScale.cxx
 * \page niftkShiftScale
 * \section niftkShiftScaleSummary Runs the ITK ShiftScaleImageFilter.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs the ITK ShiftScaleImageFilter." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;   
    std::cout << "    -shift <short> 0        Shift value" << std::endl;
    std::cout << "    -scale <short> 1        Scale value" << std::endl;
  }

/**
 * \brief Takes image and does shifting/scaling in ITK style.
 */
int main(int argc, char** argv)
{

  const   unsigned int Dimension = 3;
  typedef short        PixelType;

  // Define command line params
  std::string inputImage;
  std::string outputImage;
  float shift = 0;
  float scale = 1;
  

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
    else if(strcmp(argv[i], "-shift") == 0){
      shift=atof(argv[++i]);
      std::cout << "Set -shift=" << niftk::ConvertToString(shift) << std::endl;
    }
    else if(strcmp(argv[i], "-scale") == 0){
      scale=atof(argv[++i]);
      std::cout << "Set -scale=" << niftk::ConvertToString(scale) << std::endl;
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
  typedef itk::ShiftScaleImageFilter<InputImageType, InputImageType> ShiftScaleFilterType;
  
  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(inputImage);
  
  ShiftScaleFilterType::Pointer filter = ShiftScaleFilterType::New();
  filter->SetInput(imageReader->GetOutput());
  filter->SetShift(shift);
  filter->SetScale(scale);
  
  std::cout << "Filtering with shift:" << niftk::ConvertToString(shift)
      << ", scale:" << niftk::ConvertToString(scale) << std::endl;
  
  OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
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
