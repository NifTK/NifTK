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
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMultiplyImageFilter.h>
#include <itkExtendedBrainMaskWithSmoothDropOffCompositeFilter.h>

/*!
 * \file niftkDilateMaskAndCrop.cxx
 * \page niftkDilateMaskAndCrop
 * \section niftkDilateMaskAndCropSummary Takes an image and a mask, and crops the image around the mask.
 *
 * This program was designed for Nicky Hobbs to create a brain mask for her 
 * fluid registration experiments. As per email from Jo Barnes, you have 4 parameters,
 * 
 * <pre>
 * 
 * 1.) Threshold using value T
 * 2.) First stage of dilation = X iterations, using a radius 1 kernel
 * 3.) Second stage of dilation = Y iterations, using a radius 1 kernel
 * 4.) Gaussian FWHM = Z millimetres
 * 
 * </pre>
 * 
 * and the process is:
 * 
 * <pre>
 * 
 * 1.) Threshold using value T so that if value < T, output = 0 else value = 1.
 * 2.) Dilate mask by X iterations in all directions, let's call the result mask A (binary);
 * 3.) Dilate mask A further by Y iterations in all directions, let's call the result mask B (binary);
 * 4.) Applied to mask B gaussian smoothing with a Zmm FWHM kernel, to generate an approximation 
 * to a smooth signal drop-off, let's call the result mask C (reals);
 * 5.) Replaced the inner portion of mask C with mask A, i.e. to enforce the preservation of the 
 * original signal intensity values in a neighbourhood of 8 voxels around the original brain mask, 
 * followed by a smooth drop-off around that, let's call the result mask D (reals);
 * 6.) Then multiply the image by the mask.
 * 
 * </pre>
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs Fluid Cropping process" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -m maskFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image " << std::endl;
    std::cout << "    -m    <filename>        Mask image " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -t    <float> [127]     Initial threshold for mask" << std::endl;
    std::cout << "    -f    <int>   [3]       First round of dilations" << std::endl;
    std::cout << "    -s    <int>   [2]       Second round of dilations" << std::endl;
    std::cout << "    -g    <float> [2]       Gaussian FWHM in mm" << std::endl;
  }

int main(int argc, char** argv)
{

  const   unsigned int Dimension = 3;
  typedef float        PixelType;

  // Define command line params
  std::string input;
  std::string mask;
  int firstDilations = 3;
  int secondDilations = 2;
  double fwhm = 2;
  double threshold = 127;   
  std::string output;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      input=argv[++i];
      std::cout << "Set -i=" << input << std::endl;
    }
    else if(strcmp(argv[i], "-m") == 0){
      mask=argv[++i];
      std::cout << "Set -m=" << mask << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      output=argv[++i];
      std::cout << "Set -o=" << output << std::endl;
    }
    else if(strcmp(argv[i], "-f") == 0){
      firstDilations=atoi(argv[++i]);
      std::cout << "Set -f=" << niftk::ConvertToString(firstDilations) << std::endl;
    }
    else if(strcmp(argv[i], "-s") == 0){
      secondDilations=atoi(argv[++i]);
      std::cout << "Set -s=" << niftk::ConvertToString(secondDilations) << std::endl;
    }
    else if(strcmp(argv[i], "-g") == 0){
      fwhm=atof(argv[++i]);
      std::cout << "Set -g=" << niftk::ConvertToString(fwhm) << std::endl;
    }
    else if(strcmp(argv[i], "-t") == 0){
      threshold=atof(argv[++i]);
      std::cout << "Set -t=" << niftk::ConvertToString(threshold) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (input.length() == 0 || output.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  typedef itk::Image< PixelType, Dimension >     InputImageType;   
  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  typedef itk::ExtendedBrainMaskWithSmoothDropOffCompositeFilter< InputImageType > FilterType;
  typedef itk::MultiplyImageFilter<InputImageType, InputImageType> MultiplyFilterType;
  
  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(input);

  InputImageReaderType::Pointer maskReader = InputImageReaderType::New();
  maskReader->SetFileName(mask);

  // Create filter that generates the extended mask
  FilterType::Pointer maskFilter = FilterType::New();
  
  maskFilter->SetInput(maskReader->GetOutput());
  maskFilter->SetInitialThreshold(threshold);
  maskFilter->SetFirstNumberOfDilations(firstDilations);
  maskFilter->SetSecondNumberOfDilations(secondDilations);
  maskFilter->SetGaussianFWHM(fwhm);

  MultiplyFilterType::Pointer multiplyFilter = MultiplyFilterType::New();
  multiplyFilter->SetInput1(imageReader->GetOutput());
  multiplyFilter->SetInput2(maskFilter->GetOutput());
  
  OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  imageWriter->SetFileName(output);
  imageWriter->SetInput(multiplyFilter->GetOutput());
  
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
