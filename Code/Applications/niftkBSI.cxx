#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include "itkLogHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIntensityNormalisationCalculator.h"
#include "itkBoundaryShiftIntegralCalculator.h"
#include "itkIndent.h"
#include <stdio.h>


int main(int argc, char* argv[])
{
	if (argc < 13)
	{
	  niftk::itkLogHelper::PrintCommandLineHeader(std::cerr);
	  std::cerr << std::endl;
    std::cerr << "Classic BSI" << std::endl << std::endl; 
		std::cerr << "Program to calculate the boundary shift integral, based on the paper" << std::endl; 
		std::cerr << "  Freeborough PA and Fox NC, The boundary shift integral: an accurate and" << std::endl; 
		std::cerr << "  robust measure of cerebral volume changes from registered repeat MRI," << std::endl; 
		std::cerr << "  IEEE Trans Med Imaging. 1997 Oct;16(5):623-9." << std::endl << std::endl;
		std::cerr << "Usage: " << argv[0] << std::endl;
		std::cerr << "         <baseline image for intensity normalisation>" << std::endl; 
		std::cerr << "         <baseline mask for intensity normalisation>" << std::endl; 
		std::cerr << "         <repeat image for intensity normalisation>" << std::endl; 
		std::cerr << "         <repeat mask for intensity normalisation>" << std::endl; 
		std::cerr << "         <baseline image for BSI>" << std::endl;
		std::cerr << "         <baseline mask for BSI>" << std::endl; 
		std::cerr << "         <repeat image for BSI>" << std::endl;
		std::cerr << "         <repeat mask for BSI>" << std::endl;
		std::cerr << "         <number of erosion>" << std::endl;
		std::cerr << "         <number of dilation>" << std::endl;
		std::cerr << "         <lower intensity in the BSI window in %% mean brain intensity>" << std::endl;
		std::cerr << "         <upper intensity in the BSI window in %% mean brain intensity>" << std::endl;
		std::cerr << "         <optional: sub ROI to intersect with the BSI XOR region>" << std::endl << std::endl;
		std::cerr << "Notice that all the images and masks for intensity normalisation must " << std::endl;
		std::cerr << "have the SAME voxel sizes and image dimensions. The same applies to the " << std::endl;
		std::cerr << "images and masks for BSI." << std::endl;
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


