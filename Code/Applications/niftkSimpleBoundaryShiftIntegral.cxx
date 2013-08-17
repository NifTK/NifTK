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
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkIntensityNormalisationCalculator.h>
#include <itkBoundaryShiftIntegralCalculator.h>
#include <itkSimpleKMeansClusteringImageFilter.h>
#include <itkBinariseUsingPaddingImageFilter.h>
#include <itkIndent.h>
#include <stdio.h>

/*!
 * \file niftkSimpleBoundaryShiftIntegral.cxx
 * \page niftkSimpleBoundaryShiftIntegral
 * \section niftkSimpleBoundaryShiftIntegralSummary     Program to calculate the boundary shift integral, based on the paper". 
 * 
 * Program to calculate the boundary shift integral, based on the paper
 * Freeborough PA and Fox NC, The boundary shift integral: an accurate and robust measure of cerebral volume changes from registered repeat MRI,
 * IEEE Trans Med Imaging. 1997 Oct;16(5):623-9.
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float, double
 *
 * \section niftkSimpleBoundaryShiftIntegralCaveat Caveats
 * \li Notice that all the images and masks for intensity normalisation must have the SAME voxel sizes and image dimensions. The same applies to the images and masks for BSI.
 */

int main(int argc, char* argv[])
{
	try
	{
		typedef itk::Image<double, 3> DoubleImageType;
		typedef itk::Image<int, 3> IntImageType;

		typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
		typedef itk::ImageFileReader<IntImageType> IntReaderType;
		typedef itk::ImageFileWriter<IntImageType> WriterType;
		typedef itk::BoundaryShiftIntegralCalculator<DoubleImageType,IntImageType,IntImageType> BoundaryShiftIntegralFilterType;

		BoundaryShiftIntegralFilterType::Pointer bsiFilter = BoundaryShiftIntegralFilterType::New();
		WriterType::Pointer writer = WriterType::New();
		DoubleReaderType::Pointer baselineBSIImageReader = DoubleReaderType::New();
		DoubleReaderType::Pointer repeatBSIImageReader = DoubleReaderType::New();
		IntReaderType::Pointer baselineBSIMaskReader = IntReaderType::New();
		IntReaderType::Pointer repeatBSIMaskReader = IntReaderType::New();
		IntReaderType::Pointer subROIMaskReader = IntReaderType::New();

		baselineBSIImageReader->SetFileName(argv[1]);
		baselineBSIMaskReader->SetFileName(argv[2]);
		repeatBSIImageReader->SetFileName(argv[3]);
		repeatBSIMaskReader->SetFileName(argv[4]);

		bsiFilter->SetBaselineImage(baselineBSIImageReader->GetOutput());
		bsiFilter->SetBaselineMask(baselineBSIMaskReader->GetOutput());
		bsiFilter->SetRepeatImage(repeatBSIImageReader->GetOutput());
		bsiFilter->SetRepeatMask(repeatBSIMaskReader->GetOutput());
		bsiFilter->SetBaselineIntensityNormalisationFactor(atof(argv[5]));
		bsiFilter->SetRepeatIntensityNormalisationFactor(atof(argv[6]));
		bsiFilter->SetNumberOfErosion(atoi(argv[7]));
		bsiFilter->SetNumberOfDilation(atoi(argv[8]));
		bsiFilter->SetLowerCutoffValue(atof(argv[9]));
		bsiFilter->SetUpperCutoffValue(atof(argv[10]));
		bsiFilter->Compute();
		std::cout << argv[1] << "," << argv[2] << "," << argv[3] << "," << argv[4] << ",BSI," << bsiFilter->GetBoundaryShiftIntegral() << std::endl;
	}
	catch (itk::ExceptionObject& itkException)
	{
		std::cerr << "Error: " << itkException << std::endl;
		return EXIT_FAILURE;
	}
  
  return EXIT_SUCCESS;
}


