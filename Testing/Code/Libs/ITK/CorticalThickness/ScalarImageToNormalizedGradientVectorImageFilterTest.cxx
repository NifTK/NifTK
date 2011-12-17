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

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkImage.h"
#include "itkVector.h"
#include "itkScalarImageToNormalizedGradientVectorImageFilter.h"

/**
 * Test the normalized gradient vector generator.
 */
int ScalarImageToNormalizedGradientVectorImageFilterTest(int, char* []) 
{
  const unsigned int Dimension = 2;

  typedef double PixelType;
  typedef itk::Image<PixelType, Dimension>  InputImageType;

  typedef InputImageType::IndexType    IndexType;
  typedef InputImageType::SizeType     SizeType;
  typedef InputImageType::SpacingType  SpacingType;
  typedef InputImageType::RegionType   RegionType;
  typedef InputImageType::Pointer   ImageTypePointer;

  // Create an image
  ImageTypePointer inputImage = InputImageType::New();
  
  // Define their size, and start index
  SizeType size;
  size[0] = 4;
  size[1] = 4;

  IndexType index;
  index[0] = 0;
  index[1] = 0;

  RegionType region;
  region.SetIndex( index );
  region.SetSize( size );
  
  SpacingType spacing;
  spacing[0] = 1.1;
  spacing[1] = 1.2;
  
  // Initialize Image 
  inputImage->SetLargestPossibleRegion( region );
  inputImage->SetBufferedRegion( region );
  inputImage->SetRequestedRegion( region );
  inputImage->SetSpacing(spacing);
  inputImage->Allocate();
  
  // 10 14 14 11
  // 11 11 14 11
  // 11 15 16 17
  // 15 16 15 18
  index[0] = 0; 
  index[1] = 0; 
  inputImage->SetPixel(index, 10);                      
  index[0] = 1; 
  index[1] = 0; 
  inputImage->SetPixel(index, 14);                      
  index[0] = 2; 
  index[1] = 0; 
  inputImage->SetPixel(index, 14);                      
  index[0] = 3; 
  index[1] = 0; 
  inputImage->SetPixel(index, 11);                      
  index[0] = 0; 
  index[1] = 1; 
  inputImage->SetPixel(index, 11);                      
  index[0] = 1; 
  index[1] = 1; 
  inputImage->SetPixel(index, 11);                      
  index[0] = 2; 
  index[1] = 1; 
  inputImage->SetPixel(index, 14);                      
  index[0] = 3; 
  index[1] = 1; 
  inputImage->SetPixel(index, 11);                      
  index[0] = 0; 
  index[1] = 2; 
  inputImage->SetPixel(index, 15);                      
  index[0] = 1; 
  index[1] = 2; 
  inputImage->SetPixel(index, 16);                      
  index[0] = 2; 
  index[1] = 2; 
  inputImage->SetPixel(index, 17);           
  index[0] = 3; 
  index[1] = 2; 
  inputImage->SetPixel(index, 11);                      
  index[0] = 0; 
  index[1] = 3; 
  inputImage->SetPixel(index, 15);                      
  index[0] = 1; 
  index[1] = 3; 
  inputImage->SetPixel(index, 16);                      
  index[0] = 2; 
  index[1] = 3; 
  inputImage->SetPixel(index, 15);                      
  index[0] = 3; 
  index[1] = 3; 
  inputImage->SetPixel(index, 18);                     
  
  typedef itk::ScalarImageToNormalizedGradientVectorImageFilter<InputImageType, float> FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(inputImage);
  filter->SetNumberOfThreads(1);
  filter->Update();
  
  // Check output
  double result;
  index[0] = 1;
  index[1] = 1;
  result = filter->GetOutput()->GetPixel(index)[0];
  if (fabs(result - 0.853282) > 0.0001) return EXIT_FAILURE;
  
  index[0] = 2;
  index[1] = 1;
  result = filter->GetOutput()->GetPixel(index)[0];
  if (fabs(result - 0      ) > 0.0001) return EXIT_FAILURE;
  
  index[0] = 1;
  index[1] = 2;
  result = filter->GetOutput()->GetPixel(index)[0];
  if (fabs(result - 0.399944) > 0.0001) return EXIT_FAILURE;

  index[0] = 2;
  index[1] = 2;
  result = filter->GetOutput()->GetPixel(index)[0];
  if (fabs(result - -0.983607) > 0.0001) return EXIT_FAILURE;

  index[0] = 1;
  index[1] = 1;
  result = filter->GetOutput()->GetPixel(index)[1];
  if (fabs(result - 0.52145) > 0.0001) return EXIT_FAILURE;

  index[0] = 2;
  index[1] = 1;
  result = filter->GetOutput()->GetPixel(index)[1];  
  if (fabs(result - 1    ) > 0.0001) return EXIT_FAILURE;

  index[0] = 1;
  index[1] = 2;  
  result = filter->GetOutput()->GetPixel(index)[1];
  if (fabs(result - 0.916539 ) > 0.0001) return EXIT_FAILURE;

  index[0] = 2;
  index[1] = 2;   
  result = filter->GetOutput()->GetPixel(index)[1]; 
  if (fabs(result - 0.180328) > 0.0001) return EXIT_FAILURE;
         
  return EXIT_SUCCESS;

}




