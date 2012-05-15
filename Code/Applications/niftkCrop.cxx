/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-21 14:43:44 +0000 (Mon, 21 Nov 2011) $
 Revision          : $Revision: 7828 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "itkCropTargetImageWhereSourceImageNonZero.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

/*!
 * \file niftkCrop.cxx
 * \page niftkCrop
 * \section niftkCropSummary Crops the input image using the mask.
 *
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of type short.
 *
 * \section niftkCropCavear Caveats
 * \li File sizes not checked.
 * \li Image headers not checked. By "voxel by voxel basis" we mean that the image geometry, origin, orientation is not checked.
 */

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Crops the input image using the mask." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -m maskFileName -o outputFileName" << std::endl;
    std::cout << "  " << std::endl;
    return;
  }

/**
 * \brief Takes mask and target and crops target where mask is non zero.
 */
int main(int argc, char** argv)
{

  const   unsigned int Dimension = 3;
  typedef short        PixelType;
  
  // Define command line args
  std::string inputImage;
  std::string maskImage;
  std::string outputImage;
  

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
      std::cout << "Set -m=" << maskImage << std::endl;
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

  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  InputImageReaderType::Pointer maskReader = InputImageReaderType::New();
  OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  
  imageReader->SetFileName(inputImage);
  maskReader->SetFileName(maskImage);
  
  itk::CropTargetImageWhereSourceImageNonZeroImageFilter<InputImageType, InputImageType>::Pointer filter 
    = itk::CropTargetImageWhereSourceImageNonZeroImageFilter<InputImageType, InputImageType>::New();  
  filter->SetInput1(maskReader->GetOutput());
  filter->SetInput2(imageReader->GetOutput());
  
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
