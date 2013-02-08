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
#include "itkVector.h"
#include "itkImage.h"
#include "itkGaussianSmoothVectorFieldFilter.h"

/**
 * Test the GaussianSmoothVectorFieldFilter.
 */
int GaussianSmoothVectorFieldFilterTest(int argc, char * argv[])
{
	// Simple test to see if we can do 4D filtering!
	// In practice, this only works with > 5 voxels in each dimension.

	typedef float PixelType;
	typedef itk::Vector<PixelType, 3> VectorPixelType;
	typedef itk::Image<VectorPixelType, 4> TimeVaryingVelocityImageType;
	typedef TimeVaryingVelocityImageType::PixelType TimeVaryingVelocityPixelType;
	typedef itk::GaussianSmoothVectorFieldFilter<float, 4, 3> FilterType;

	// Index, as always is (x,y,z,t) = (0,0,0,0)
	TimeVaryingVelocityImageType::IndexType index;
	index.Fill(0);

	// Initialize vector with (x,y,z) = (0,0,0)
	TimeVaryingVelocityImageType::PixelType pixel;
	pixel.Fill(1);

	// Size will be 7x7x7x2
	TimeVaryingVelocityImageType::SizeType size;
	size.Fill(7);
	size[3] = 2;

	// Create image, and fill with zero.
	TimeVaryingVelocityImageType::RegionType region;
	region.SetSize(size);
	region.SetIndex(index);
	TimeVaryingVelocityImageType::Pointer inputImage = TimeVaryingVelocityImageType::New();
	inputImage->SetRegions(region);
	inputImage->Allocate();
	inputImage->FillBuffer(pixel);

	// Create a single bright pixel.
	index[0] = 3;
	index[1] = 3;
	index[2] = 3;
	index[3] = 0;
	pixel.Fill(100);
	inputImage->SetPixel(index, pixel);

	// Sigma is 1 in all directions, except time.
	FilterType::SigmaType sigma;
	sigma.Fill(1);
	sigma[3] = 0;

	// Filter image;
	FilterType::Pointer filter = FilterType::New();
	filter->SetInput(inputImage);
	filter->SetSigma(sigma);
	filter->SetInPlace(true);
	filter->Update();

	for (unsigned int t = 0; t < 2; t++)
	{
		for (unsigned int z = 0; z < 7; z++)
		{
			for (unsigned int y = 0; y < 7; y++)
			{
				for (unsigned int x = 0; x < 7; x++)
				{
					index[0] = x;
					index[1] = y;
					index[2] = z;
					index[3] = t;

					std::cout << "Index=" << index << ", pixel=" << filter->GetOutput()->GetPixel(index) << std::endl;
				}
			}
		}
	}

	// Better test something, or else its not a test.. its just code.
	// At least it functions as a regression test then.

	index[0] = 3;
	index[1] = 3;
	index[2] = 3;
	index[3] = 0;

	pixel = filter->GetOutput()->GetPixel(index);
	if (fabs(pixel[0] - 11.07) > 0.0001)
	  {
	    std::cout << "Expected 11.07, but got:" << pixel[0] << std::endl;
	    return EXIT_FAILURE;
	  }
	if (fabs(pixel[1] - 11.07) > 0.0001)
	  {
	    std::cout << "Expected 11.07, but got:" << pixel[1] << std::endl;
	    return EXIT_FAILURE;
	  }
	if (fabs(pixel[2] - 11.07) > 0.0001)
	  {
	    std::cout << "Expected 11.07, but got:" << pixel[2] << std::endl;
	    return EXIT_FAILURE;
	  }
	return EXIT_SUCCESS;
}
