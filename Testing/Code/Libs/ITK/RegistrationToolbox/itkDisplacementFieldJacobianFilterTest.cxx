/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include "itkDisplacementFieldJacobianFilter.h"

int DisplacementFieldJacobianFilterTest(int argc, char * argv[])
{
  const unsigned int Dimension = 2;
  typedef itk::Vector<short,2> PixelType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::Index<Dimension>  IndexType;
  typedef itk::Size<Dimension> SizeType;
  typedef itk::ImageRegion<Dimension> RegionType;

  ImageType::Pointer inputImage = ImageType::New();
  
  SizeType size;
  size[0] = 2;
  size[1] = 2;
  
  IndexType start;
  start[0] = 0;
  start[1] = 0;

  RegionType region;
  region.SetIndex(start);
  region.SetSize(size);

  // Initialize Image A
  inputImage->SetRegions(region);
  inputImage->Allocate();
  
  PixelType value;
  
  // [0,0]  [1,2]
  // [2,1]  [2,2]
  start[0] = 0;
  start[1] = 0;
  value[0] = 0; 
  value[1] = 0; 
  inputImage->SetPixel(start, value); 
  start[0] = 1;
  start[1] = 0;
  value[0] = 1; 
  value[1] = 2; 
  inputImage->SetPixel(start, value); 
  start[0] = 0;
  start[1] = 1;
  value[0] = 2; 
  value[1] = 1; 
  inputImage->SetPixel(start, value); 
  start[0] = 1;
  start[1] = 1;
  value[0] = 2; 
  value[1] = 2; 
  inputImage->SetPixel(start, value); 
  
  typedef itk::DisplacementFieldJacobianFilter<short, float, Dimension> FilterType;
  FilterType::Pointer filter = FilterType::New(); 
  filter->SetInput(inputImage); 
  filter->Update(); 
  
  for (unsigned int i = 0; i < Dimension; i++)
  {
    for (unsigned int j = 0; j < Dimension; j++)
    {
      start[0] = i; 
      start[1] = j; 
      std::cout << inputImage->GetPixel(start) << std::endl; 
      std::cout << filter->GetOutput()->GetPixel(start) << std::endl; 
    }
  }
      
  start[0] = 0;
  start[1] = 0;
  FilterType::OutputImageType::PixelType matrix = filter->GetOutput()->GetPixel(start); 
  if (matrix(0,0) != 1.5 && matrix(1,0) != 1. && matrix(0,1) != 1 && matrix(1,1) != 1)
  {
    std::cout << "Failed:" << start << ":" << matrix << std::endl; 
    return EXIT_FAILURE;
  }
  start[0] = 0;
  start[1] = 1;
  matrix = filter->GetOutput()->GetPixel(start); 
  if (matrix(0,0) != 1. && matrix(1,0) != 1. && matrix(0,1) != 0.5 && matrix(1,1) != 1.5)
  {
    std::cout << "Failed:" << start << ":" << matrix << std::endl; 
    return EXIT_FAILURE;
  }
  start[0] = 1;
  start[1] = 0;
  matrix = filter->GetOutput()->GetPixel(start); 
  if (matrix(0,0) != 1.5 && matrix(1,0) != 0.5 && matrix(0,1) != 1.0 && matrix(1,1) != 0.)
  {
    std::cout << "Failed:" << start << ":" << matrix << std::endl; 
    return EXIT_FAILURE;
  }
  start[0] = 1;
  start[1] = 1;
  matrix = filter->GetOutput()->GetPixel(start); 
  if (matrix(0,0) != 1. && matrix(1,0) != 0.5 && matrix(0,1) != 0.5 && matrix(1,1) != 1.)
  {
    std::cout << "Failed:" << start << ":" << matrix << std::endl; 
    return EXIT_FAILURE;
  }
  
  std::cout << "Passed" << std::endl; 
  return EXIT_SUCCESS; 
}















