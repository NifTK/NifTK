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
#include "itkDemonsRegistrationFilter.h"

/**
 * A simple test harness for the the DemonsRegistrationFilter.
 * This is not because I think it doesn't work, its because
 * I want to understand how it works, and if I can use it as
 * a component part of a "Registration Based Cortical Thickness"
 * pipeline (Das. et. al. NeuroImage 2009).
 */
int DemonsRegistrationFilterTest(int argc, char * argv[])
{

	typedef itk::Image<float, 2> ImageType;
	typedef itk::Vector<float, 2> VectorType;
	typedef itk::Image<VectorType, 2> VectorImageType;

	unsigned int imageSize = 10;

	ImageType::SizeType size;
	size[0] = imageSize;
	size[1] = imageSize;
	ImageType::IndexType index;
	index[0] = 0;
	index[1] = 0;
	ImageType::RegionType region;
	region.SetSize(size);
	region.SetIndex(index);

	ImageType::Pointer fixedImage = ImageType::New();
	fixedImage->SetRegions(region);
	fixedImage->Allocate();
	fixedImage->FillBuffer(0);

	ImageType::Pointer movingImage = ImageType::New();
	movingImage->SetRegions(region);
	movingImage->Allocate();
	movingImage->FillBuffer(0);

	// Imagine a registration experiment, where we are
	// matching the WM binary mask image to the
	// WM+GM binary mask image. So, this experiment simulates
	// that with 2 square sections. The moving image is the
	// WM image, and the fixed image is WM+GM, and in this
	// case we are just registering rectangles.
	for (unsigned int x = 0; x < imageSize; x++)
	{
		for (unsigned int y = 0; y < imageSize; y++)
		{
			// So from 0-3, moving image is white (1), and 4-9 its black (0).
			if (x < 4)
			{
				index[0] = x;
				index[1] = y;
				movingImage->SetPixel(index, 1);
			}

			// So from 0-6, fixed image is white (1), and 7-9 its black (1).
			if (x < 7)
			{
				index[0] = x;
				index[1] = y;
				fixedImage->SetPixel(index, 1);
			}
		}
	}

	typedef itk::DemonsRegistrationFilter<ImageType, ImageType, VectorImageType> FilterType;

	FilterType::Pointer filter = FilterType::New();
	filter->SetFixedImage(fixedImage);
	filter->SetMovingImage(movingImage);
	filter->SetIntensityDifferenceThreshold(0.001);
	filter->SetSmoothUpdateField(false);
	filter->SetUpdateFieldStandardDeviations(1);
	filter->SetSmoothDeformationField(false);
	filter->SetStandardDeviations(1);
	filter->SetNumberOfIterations(1);
	filter->Update();

	// So, if:
	// Fixed image =  1 1 1 1 1 1 1 0 0 0,
	// Moving image = 1 1 1 1 0 0 0 0 0 0
	// The transformation we require is to warp moving image into reference frame of
	// fixed image, so there should be gradient in columns 4, 5 and 6, driving to the left (negative).
    // Gradient in pixel 5 is optional, depending on how you do finite diff.
    // Gradient in pixel 4, 6 depends on gradient type. If Symmetric, you get both,
	// if fixed, you just get fixed, if MappedMoving or WarpedMoving you just get moving

	VectorType def;
	def[0] = -1;
	def[1] = 0;

	VectorImageType::Pointer initialDeformation = VectorImageType::New();
	initialDeformation->SetRegions(region);
	initialDeformation->Allocate();
	initialDeformation->FillBuffer(def);

	filter->SetInitialDeformationField(initialDeformation);
	filter->Update();

	for (unsigned int y = 0; y < imageSize; y++)
	{
		for (unsigned int x = 0; x < imageSize; x++)
		{
			index[0] = x;
			index[1] = y;

			std::cout << "index=" << index << ", value=" << filter->GetOutput()->GetPixel(index) << std::endl;
		}
	}

	// Better actually test something, or else, its not a unit test.
	double tol = 0.0001;

	index[0] = 0;
	index[1] = 0;
	def = filter->GetOutput()->GetPixel(index);
	if (fabs(def[0] - -1) > tol)
	{
		std::cout << "Expected -1, but got:" << def[0] << std::endl;
		return EXIT_FAILURE;
	}
	if (fabs(def[1] - 0) > tol)
	{
		std::cout << "Expected 0, but got:" << def[1] << std::endl;
		return EXIT_FAILURE;
	}

	index[0] = 6;
	index[1] = 0;
	def = filter->GetOutput()->GetPixel(index);
	if (fabs(def[0] - -1.4) > tol)
	{
		std::cout << "Expected -1.4, but got:" << def[0] << std::endl;
		return EXIT_FAILURE;
	}
	if (fabs(def[1] - 0) > tol)
	{
		std::cout << "Expected 0, but got:" << def[1] << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
