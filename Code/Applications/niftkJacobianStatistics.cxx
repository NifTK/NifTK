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
#include "itkImageRegionConstIterator.h"

/*!
 * \file niftkJacobianStatistics.cxx
 * \page niftkJacobianStatistics
 * \section niftkJacobianStatisticsSummary Take a jacobian image, and a mask, and calculate statistics for the region.
 *
 * This program take a jacobian image, and a mask, and calculate statistics for the region.
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float
 *
 */

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes a jacobian image and a mask and calculates statistics for the masked region" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputJacobianFileName -m maskImageName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input Jacobian image " << std::endl;
    std::cout << "    -m    <filename>        Mask image " << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;   
  }

/**
 * \brief Take a jacobian image, and a mask, and calculate statistics for the region.
 */
int main(int argc, char** argv)
{
  const   unsigned int Dimension = 3;
  typedef float        PixelType;

  // Define command line params
  std::string inputImage;
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
  if (inputImage.length() == 0 || maskImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  typedef itk::Image< PixelType, Dimension >     InputImageType;   
  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;

  InputImageReaderType::Pointer  jacobianReader  = InputImageReaderType::New();
  InputImageReaderType::Pointer  maskReader  = InputImageReaderType::New();
  
  jacobianReader->SetFileName( inputImage );
  maskReader->SetFileName( maskImage );
  
  try
  {
    jacobianReader->Update(); 
    maskReader->Update();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cout << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  if (jacobianReader->GetOutput()->GetLargestPossibleRegion().GetSize()
      != maskReader->GetOutput()->GetLargestPossibleRegion().GetSize())
    {
      std::cout << "Jacobian image and mask image must be the same size" << std::endl; 
      return EXIT_FAILURE;
    }
  
  itk::ImageRegionConstIterator< InputImageType > jacobianImageIterator(jacobianReader->GetOutput(), jacobianReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator< InputImageType > maskImageIterator(maskReader->GetOutput(), maskReader->GetOutput()->GetLargestPossibleRegion());
  
  double sum = 0;
  double mean = 0;
  unsigned long int counter = 0;
  
  jacobianImageIterator.GoToBegin();
  maskImageIterator.GoToBegin();
  
  while(!jacobianImageIterator.IsAtEnd() && !maskImageIterator.IsAtEnd())
    {
      if (maskImageIterator.Get() > 0)
        {
          sum += jacobianImageIterator.Get();
          mean += jacobianImageIterator.Get();
          counter++;
        }
      ++jacobianImageIterator;
      ++maskImageIterator;
    }
  mean /= (double)counter;
  
  std::cout << "Jacobian image=" <<  inputImage << ", mask image=" << maskImage << ", sum=" << sum << ", mean=" << mean << ", voxels=" << counter << std::endl;
  return EXIT_SUCCESS; 
}
