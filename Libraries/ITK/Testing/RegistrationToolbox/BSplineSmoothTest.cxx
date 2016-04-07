/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include <memory>
#include <math.h>
#include <itkVector.h>
#include <itkImage.h>
#include <itkBSplineSmoothVectorFieldFilter.h>
#include <itkBSplineOperator.h>

/**
 * Basic tests for BSpline smoothing.
 */
int BSplineSmoothTest(int argc, char * argv[])
{

  // Define the dimension of the images
  const unsigned int Dimension = 2;

  // Declare the types of the images
  typedef double DataType;
  typedef itk::Vector<DataType, Dimension>                         PixelType;
  typedef itk::Image<PixelType, Dimension>                         ImageType;
  typedef itk::BSplineSmoothVectorFieldFilter<DataType, Dimension> BSplineSmootherType;
  typedef BSplineSmootherType::GridSpacingType                     GridSpacingType;
  typedef itk::BSplineOperator<DataType, Dimension>                BSplineOperatorType;
  
  // Create an image. 
  ImageType::Pointer image = ImageType::New();
  typedef itk::Index<Dimension>  IndexType;
  typedef itk::Size<Dimension> SizeType;
  typedef itk::ImageRegion<Dimension> RegionType;
  
  PixelType pixel;
  pixel[0] = 0;
  pixel[1] = 0;
  
  SizeType size;
  size[0] = 10;
  size[1] = 10;

  IndexType index;
  index[0] = 0;
  index[1] = 0;

  RegionType region;
  region.SetIndex( index );
  region.SetSize( size );
  
  image->SetLargestPossibleRegion( region );
  image->SetBufferedRegion( region );
  image->SetRequestedRegion( region );
  image->Allocate();

  // Put some data in.
  for (unsigned int i = 0; i < size[0]; i++)
    {
      for (unsigned int j = 0; j < size[1]; j++)
        {
          index[0] = i;
          index[1] = j;
          pixel[0] = 0;
          pixel[1] = 0;
          image->SetPixel(index, pixel);
        }
    }
  
  // This will be like testing smoothing a spike.
  index[0] = 5;
  index[1] = 5;
  pixel[0] = 10;
  pixel[1] = 10;
  image->SetPixel(index, pixel);

  // Set the grid spacing.
  GridSpacingType spacing;
  spacing[0] = 2;
  spacing[1] = 2;
  
  BSplineSmootherType::Pointer smoother = BSplineSmootherType::New();
  smoother->SetInput(image);
  smoother->SetGridSpacing(spacing);
  smoother->Update();
  
  ImageType::Pointer output = smoother->GetOutput();
  SizeType outputSize = output->GetLargestPossibleRegion().GetSize();
  
  // Check that same size image comes out.
  if (size[0] != outputSize[0]) return EXIT_FAILURE;
  if (size[1] != outputSize[1]) return EXIT_FAILURE;
  
  // Put some data in.
  double totalX = 0;
  for (unsigned int i = 0; i < outputSize[0]; i++)
    {
      for (unsigned int j = 0; j < outputSize[1]; j++)
        {
          index[0] = i;
          index[1] = j;
          std::cerr << "Output:[" << i << "," << j << "]=" << output->GetPixel(index) << std::endl;
          totalX += output->GetPixel(index)[0];
        }
    }

  // Initially, the BSpline Smoothing was set, so the kernel summed to 1.
  // However, Marc's kernels do not sum to 1.
  if (fabs(totalX - 39.999999999999993) > 0.000001) return EXIT_FAILURE;
  
  // And then check the middle value.
  index[0] = 5;
  index[1] = 5;
  double middleX = output->GetPixel(index)[0];
  
  if (fabs(middleX - 4.44444444444444) > 0.000001) return EXIT_FAILURE;
  
  // We are done. Go for coffee.
  return EXIT_SUCCESS;    
}
