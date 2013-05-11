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
#include <itkInterpolateVectorFieldFilter.h>

/**
 * Basic tests for BSpline interpolating.
 * Once we have a vector image per voxel, we smooth it, and then interpolate it at each grid point.
 * This class tests the interpolation at each grid point.
 */
int BSplineInterpolateTest(int argc, char * argv[])
{

  // Define the dimension of the images
  const unsigned int Dimension = 2;

  // Declare the types of the images
  typedef double DataType;
  typedef itk::Vector<DataType, Dimension>                         PixelType;
  typedef itk::Image<PixelType, Dimension>                         ImageType;
  typedef itk::InterpolateVectorFieldFilter<DataType, Dimension>   InterpolateFilterType;
  
  // Create an image of vectors to model the deformation per voxel.
  ImageType::Pointer image = ImageType::New();
  typedef itk::Index<Dimension>  IndexType;
  typedef itk::Size<Dimension> SizeType;
  typedef itk::ImageRegion<Dimension> RegionType;
  typedef ImageType::SpacingType SpacingType;
  typedef ImageType::DirectionType DirectionType;
  typedef ImageType::PointType OriginType;
  
  SizeType size;
  size[0] = 4;
  size[1] = 4;

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

  PixelType pixel;
  pixel[0] = 0;
  pixel[1] = 0;
  
  // Put some data in.
  for (unsigned int i = 0; i < size[0]; i++)
    {
      for (unsigned int j = 0; j < size[1]; j++)
        {
          index[0] = i;
          index[1] = j;
          pixel[0] = i;
          pixel[1] = size[1] - j - 1;
          image->SetPixel(index, pixel);
          std::cerr << "Vector field [" << i << "," << j << "]=" << image->GetPixel(index) << std::endl;
        }
    }
  
  // Now create a smaller image to simulate the grid.
  ImageType::Pointer grid = ImageType::New();

  size[0] = 3;
  size[1] = 3;

  index[0] = 0;
  index[1] = 0;

  region.SetIndex( index );
  region.SetSize( size );
  
  OriginType origin;
  origin[0] = 0.5;
  origin[1] = 0.75;
  
  SpacingType spacing;
  spacing[0] = 1;
  spacing[1] = 1;
  
  grid->SetSpacing(spacing);
  grid->SetOrigin(origin);
  grid->SetLargestPossibleRegion( region );
  grid->SetBufferedRegion( region );
  grid->SetRequestedRegion( region );
  grid->Allocate();

  InterpolateFilterType::Pointer interpolator = InterpolateFilterType::New();
  interpolator->SetInterpolatedField(image);
  interpolator->SetInterpolatingField(grid);
  interpolator->Update();
  
  ImageType::Pointer output = interpolator->GetOutput();
  SizeType outputSize = output->GetLargestPossibleRegion().GetSize();
  
  // Check that right size image comes out.
  if (size[0] != outputSize[0]) return EXIT_FAILURE;
  if (size[1] != outputSize[1]) return EXIT_FAILURE;
  
  // dump image.
  for (unsigned int i = 0; i < outputSize[0]; i++)
    {
      for (unsigned int j = 0; j < outputSize[1]; j++)
        {
          index[0] = i;
          index[1] = j;
          std::cerr << "Output:[" << i << "," << j << "]=" << output->GetPixel(index) << std::endl;
          
          if (fabs(output->GetPixel(index)[0] - (((double)i) + 0.5)) > 0.0000001) return EXIT_FAILURE;
           
          if (fabs(output->GetPixel(index)[1] - ( size[1] - ((double)j) - 1.0 + 0.25)) > 0.0000001) return EXIT_FAILURE;
        }
    }

  // We are done. Go for coffee.
  return EXIT_SUCCESS;    
}
