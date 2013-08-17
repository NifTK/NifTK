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
#include <itkLogHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkIntensityNormalisationCalculator.h>
#include <itkBoundaryShiftIntegralCalculator.h>
#include <itkIndent.h>
#include <stdio.h>

/*!
 * \file niftkBSI.cxx
 * \page niftkBSI
 * \section niftkBSISummary Program to calculate the boundary shift integral, based on the paper".
 * 
 * Program to calculate the boundary shift integral, based on the paper
 * Freeborough PA and Fox NC, The boundary shift integral: an accurate and robust measure of cerebral volume changes from registered repeat MRI,
 * IEEE Trans Med Imaging. 1997 Oct;16(5):623-9.
 *
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float, double
 *
 * \section niftkBSICaveat Caveats
 * \li Notice that all the images and masks for intensity normalisation must have the SAME voxel sizes and image dimensions. The same applies to the images and masks for BSI.
 */
int main(int argc, char* argv[])
{
	if (argc < 13)
	{
	  niftk::itkLogHelper::PrintCommandLineHeader(std::cerr);
	  std::cout << std::endl;
    std::cout << "Classic BSI. Program to calculate the boundary shift integral, based on the paper: Freeborough PA and Fox NC, The boundary shift integral: an accurate and robust measure of cerebral volume changes from registered repeat MRI, IEEE Trans Med Imaging. 1997 Oct;16(5):623-9." << std::endl;
		std::cout << "  " << std::endl;
		std::cout << "Usage: " << argv[0] << std::endl;
		std::cout << "         <baseline image for intensity normalisation>" << std::endl;
		std::cout << "         <baseline mask for intensity normalisation>" << std::endl;
		std::cout << "         <repeat image for intensity normalisation>" << std::endl;
		std::cout << "         <repeat mask for intensity normalisation>" << std::endl;
		std::cout << "         <baseline image for BSI>" << std::endl;
		std::cout << "         <baseline mask for BSI>" << std::endl;
		std::cout << "         <repeat image for BSI>" << std::endl;
		std::cout << "         <repeat mask for BSI>" << std::endl;
		std::cout << "         <number of erosion>" << std::endl;
		std::cout << "         <number of dilation>" << std::endl;
		std::cout << "         <lower intensity in the BSI window in %% mean brain intensity>" << std::endl;
		std::cout << "         <upper intensity in the BSI window in %% mean brain intensity>" << std::endl;
		std::cout << "         <optional: sub ROI to intersect with the BSI XOR region>" << std::endl << std::endl;
		std::cout << "Notice that all the images and masks for intensity normalisation must " << std::endl;
		std::cout << "have the SAME voxel sizes and image dimensions. The same applies to the " << std::endl;
		std::cout << "images and masks for BSI." << std::endl;
		return EXIT_FAILURE;
	}
	
	try
	{
		typedef itk::Image<double, 3> DoubleImageType;
		typedef itk::Image<int, 3> IntImageType;

		typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
		typedef itk::ImageFileReader<IntImageType> IntReaderType;
		typedef itk::ImageFileWriter<IntImageType> WriterType;
		typedef itk::IntensityNormalisationCalculator<DoubleImageType, IntImageType> IntensityNormalisationCalculatorType;
		typedef itk::BoundaryShiftIntegralCalculator<DoubleImageType,IntImageType,IntImageType> BoundaryShiftIntegralFilterType;

		DoubleReaderType::Pointer baselineNormalisationImageReader = DoubleReaderType::New();
		DoubleReaderType::Pointer repeatNormalisationImageReader = DoubleReaderType::New();
		IntReaderType::Pointer baselineNormalisationMaskReader = IntReaderType::New();
		IntReaderType::Pointer repeatNormalisationMaskReader = IntReaderType::New();

		baselineNormalisationImageReader->SetFileName(argv[1]);
		baselineNormalisationMaskReader->SetFileName(argv[2]);
		repeatNormalisationImageReader->SetFileName(argv[3]);
		repeatNormalisationMaskReader->SetFileName(argv[4]);

		IntensityNormalisationCalculatorType::Pointer normalisationCalculator = IntensityNormalisationCalculatorType::New();

		normalisationCalculator->SetInputImage1(baselineNormalisationImageReader->GetOutput());
		normalisationCalculator->SetInputImage2(repeatNormalisationImageReader->GetOutput());
		normalisationCalculator->SetInputMask1(baselineNormalisationMaskReader->GetOutput());
		normalisationCalculator->SetInputMask2(repeatNormalisationMaskReader->GetOutput());
		normalisationCalculator->Compute();
    //std::cout << "mean intensities=" << normalisationCalculator->GetNormalisationMean1() << "," 
    //                                 << normalisationCalculator->GetNormalisationMean2() << std::endl;

		BoundaryShiftIntegralFilterType::Pointer bsiFilter = BoundaryShiftIntegralFilterType::New();
		WriterType::Pointer writer = WriterType::New();
		DoubleReaderType::Pointer baselineBSIImageReader = DoubleReaderType::New();
		DoubleReaderType::Pointer repeatBSIImageReader = DoubleReaderType::New();
		IntReaderType::Pointer baselineBSIMaskReader = IntReaderType::New();
		IntReaderType::Pointer repeatBSIMaskReader = IntReaderType::New();
		IntReaderType::Pointer subROIMaskReader = IntReaderType::New();

		baselineBSIImageReader->SetFileName(argv[5]);
		baselineBSIMaskReader->SetFileName(argv[6]);
		repeatBSIImageReader->SetFileName(argv[7]);
		repeatBSIMaskReader->SetFileName(argv[8]);

		bsiFilter->SetBaselineImage(baselineBSIImageReader->GetOutput());
		bsiFilter->SetBaselineMask(baselineBSIMaskReader->GetOutput());
		bsiFilter->SetRepeatImage(repeatBSIImageReader->GetOutput());
		bsiFilter->SetRepeatMask(repeatBSIMaskReader->GetOutput());
		bsiFilter->SetBaselineIntensityNormalisationFactor(normalisationCalculator->GetNormalisationMean1());
		bsiFilter->SetRepeatIntensityNormalisationFactor(normalisationCalculator->GetNormalisationMean2());
		bsiFilter->SetNumberOfErosion(atoi(argv[9]));
		bsiFilter->SetNumberOfDilation(atoi(argv[10]));
		bsiFilter->SetLowerCutoffValue(atof(argv[11]));
		bsiFilter->SetUpperCutoffValue(atof(argv[12]));
		if (argc > 13 && strlen(argv[13])> 0)
		{
			subROIMaskReader->SetFileName(argv[13]);
			bsiFilter->SetSubROIMask(subROIMaskReader->GetOutput());
		}
		bsiFilter->Compute();
		//std::cout << "BSI=" << bsiFilter->GetBoundaryShiftIntegral() << std::endl;
		std::cout << bsiFilter->GetBoundaryShiftIntegral() << std::endl;
	}
	catch (itk::ExceptionObject& itkException)
	{
		std::cerr << "Error: " << itkException << std::endl;
		return EXIT_FAILURE;
	}
  
  return EXIT_SUCCESS;
}


