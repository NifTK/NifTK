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
#include "itkVector.h"
#include "itkImage.h"
#include "itkAddUpdateToTimeVaryingVelocityFieldFilter.h"

int AddUpdateToTimeVaryingVelocityFilterTest(int argc, char * argv[])
{

	typedef itk::Vector<float, 2> PixelType;
	typedef itk::Image<PixelType, 2> UpdateImageType;
	typedef itk::Image<PixelType, 3> VelocityImageType;
	typedef itk::AddUpdateToTimeVaryingVelocityFieldFilter<float, 2> FilterType;

	UpdateImageType::IndexType index;
	index.Fill(0);

	UpdateImageType::SizeType size;
	size.Fill(2);

	UpdateImageType::RegionType region;
	region.SetSize(size);
	region.SetIndex(index);

	PixelType pixel;
	pixel[0] = 1;
	pixel[1] = -1;

	// Update image filled with (1, -1)
	UpdateImageType::Pointer updateImage = UpdateImageType::New();
	updateImage->SetRegions(region);
	updateImage->Allocate();
	updateImage->FillBuffer(pixel);

	pixel[0] = 2;
	pixel[1] = -2;

	// Update inverse image filled with (2, -2)
	UpdateImageType::Pointer updateInverseImage = UpdateImageType::New();
	updateInverseImage->SetRegions(region);
	updateInverseImage->Allocate();
	updateInverseImage->FillBuffer(pixel);

	VelocityImageType::IndexType velocityIndex;
	velocityIndex.Fill(0);

	VelocityImageType::SizeType velocitySize;
	velocitySize.Fill(2);

	VelocityImageType::RegionType velocityRegion;
	velocityRegion.SetSize(velocitySize);
	velocityRegion.SetIndex(velocityIndex);

	pixel[0] = 4;
	pixel[1] = -4;

	// Velocity image filled with (4, -4)
	VelocityImageType::Pointer velocityImage = VelocityImageType::New();
	velocityImage->SetRegions(velocityRegion);
	velocityImage->Allocate();
	velocityImage->FillBuffer(pixel);

	double tol = 0.0001;

	// Run filter
	FilterType::Pointer filter = FilterType::New();
	filter->SetUpdateImage(updateImage);
	filter->SetInput(velocityImage);
	filter->SetTimePoint(0);
	filter->Update();

	// With one update image, and timepoint 0, only first time dimension gets updated
	velocityIndex[0] = 0;
	velocityIndex[1] = 0;
	velocityIndex[2] = 0;
	pixel = filter->GetOutput()->GetPixel(velocityIndex);
	if (fabs(pixel[0] - 5) > tol)
	{
		std::cout << "Expected 5, but got:" << pixel[0] << std::endl;
		return EXIT_FAILURE;
	}
	velocityIndex[0] = 0;
	velocityIndex[1] = 0;
	velocityIndex[2] = 1;
	pixel = filter->GetOutput()->GetPixel(velocityIndex);
	if (fabs(pixel[0] - 4) > tol)
	{
		std::cout << "Expected 4, but got:" << pixel[0] << std::endl;
		return EXIT_FAILURE;
	}

	// With two update images, and timepoint 0 the fields are added.
	// In general, the fields should be mutual exclusive.
	filter->SetUpdateImage(updateImage);
	filter->SetUpdateInverseImage(updateInverseImage);
	filter->SetInput(velocityImage);
	filter->SetTimePoint(0);
	filter->Modified();
	filter->Update();

	velocityIndex[0] = 0;
	velocityIndex[1] = 0;
	velocityIndex[2] = 0;
	pixel = filter->GetOutput()->GetPixel(velocityIndex);
	if (fabs(pixel[0] - 3) > tol)
	{
		std::cout << "Expected 3, but got:" << pixel[0] << std::endl;
		return EXIT_FAILURE;
	}
	velocityIndex[0] = 0;
	velocityIndex[1] = 0;
	velocityIndex[2] = 1;
	pixel = filter->GetOutput()->GetPixel(velocityIndex);
	if (fabs(pixel[0] - 4) > tol)
	{
		std::cout << "Expected 4, but got:" << pixel[0] << std::endl;
		return EXIT_FAILURE;
	}

	filter->SetTimePoint(1);
	filter->Modified();
	filter->Update();

	for (int t = 0; t < 2; t++)
	{
		for (int y = 0; y < 2; y++)
		{
			for (int x = 0; x < 2; x++)
			{
				index[0] = x;
				index[1] = y;

				velocityIndex[0] = x;
				velocityIndex[1] = y;
				velocityIndex[2] = t;

				std::cout << "Output: index=" << velocityIndex \
				    << ", vi=" << velocityImage->GetPixel(velocityIndex) \
				    << ", ui=" << updateImage->GetPixel(index) \
				    << ", uii=" << updateInverseImage->GetPixel(index) \
					<< ", output=" << filter->GetOutput()->GetPixel(velocityIndex) \
					<< std::endl;

			}
		}
	}

	velocityIndex[0] = 0;
	velocityIndex[1] = 0;
	velocityIndex[2] = 0;
	pixel = filter->GetOutput()->GetPixel(velocityIndex);
	if (fabs(pixel[0] - 4) > tol)
	{
		std::cout << "Expected 4, but got:" << pixel[0] << std::endl;
		return EXIT_FAILURE;
	}
	velocityIndex[0] = 0;
	velocityIndex[1] = 0;
	velocityIndex[2] = 1;
	pixel = filter->GetOutput()->GetPixel(velocityIndex);
	if (fabs(pixel[0] - 3) > tol)
	{
		std::cout << "Expected 3, but got:" << pixel[0] << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

