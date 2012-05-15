/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 17:44:42 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7864 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkInjectSourceImageGreaterThanZeroIntoTargetImageFilter.h"

/*!
 * \file niftkInject.cxx
 * \page niftkInject
 * \section niftkInjectSummary Injects a mask image into the input image.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Injects a mask image into the input image" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -m maskImageName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image " << std::endl;
    std::cout << "    -m    <filename>        Mask image " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;   
  }

/**
 * \brief Takes mask and target and injects (ie. copies values that are non zero) mask into target.
 */
int main(int argc, char** argv)
{

  const   unsigned int Dimension = 3;
  typedef short        PixelType;

  // Define command line params
  std::string inputImage;
  std::string outputImage;
  std::string maskImage;
  

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
    else if(strcmp(argv[i], "-m") == 0){
      maskImage=argv[++i];
      std::cout << "Set -v=" << maskImage << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (inputImage.length() == 0 || outputImage.length() == 0 || maskImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  typedef itk::Image< PixelType, Dimension >     InputImageType;   
  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  typedef itk::InjectSourceImageGreaterThanZeroIntoTargetImageFilter<InputImageType, InputImageType, InputImageType> InjectSourceImageGreaterThanZeroIntoTargetImageFilterType;
  
  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  InputImageReaderType::Pointer maskReader = InputImageReaderType::New();
  
  imageReader->SetFileName(inputImage);
  maskReader->SetFileName(maskImage);

  InjectSourceImageGreaterThanZeroIntoTargetImageFilterType::Pointer filter = 
    InjectSourceImageGreaterThanZeroIntoTargetImageFilterType::New();
    
  filter->SetInput1(maskReader->GetOutput());
  filter->SetInput2(imageReader->GetOutput());
  
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
