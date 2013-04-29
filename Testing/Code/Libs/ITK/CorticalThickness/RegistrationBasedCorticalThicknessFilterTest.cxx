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

#include "itkRegistrationBasedCorticalThicknessFilter.h"
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

/**
 * Test the RegistrationBasedCorticalThicknessFilter.
 */
int RegistrationBasedCorticalThicknessFilterTest(int argc, char* argv[])
{
	if (argc != 13)
	{
		std::cerr << "Usage: RegistrationBasedCorticalThicknessFilterTest wmpvImage gmwmpvImage thicknessImage boundaryImage maxIters m lambda updateSigma deformationSigma epsilon alpha outputThickness" << std::endl;
		return EXIT_FAILURE;
	}
	std::string wmpvImageFileName = argv[1];
	std::string gmwmpvImageFileName = argv[2];
	std::string thicknessImageFileName = argv[3];
	std::string boundaryImageFileName = argv[4];
	int maxIterations = atoi(argv[5]);
	int integrationSteps = atoi(argv[6]);
	float lambda = atof(argv[7]);
	float updateSigma = atof(argv[8]);
	float deformationSigma = atof(argv[9]);
	float epsilon = atof(argv[10]);
	float alpha = atof(argv[11]);
	std::string outputImage = argv[12];

	typedef float ScalarType;
	typedef itk::Vector<ScalarType, 2> VectorType;
	typedef itk::Image<ScalarType, 2> ScalarImageType;
	typedef itk::Image<VectorType, 2> VectorImageType;
	typedef itk::ImageFileReader<ScalarImageType> ScalarImageReaderType;
	typedef itk::ImageFileWriter<ScalarImageType> ScalarImageWriterType;
	typedef itk::RegistrationBasedCorticalThicknessFilter<ScalarImageType, ScalarType> FilterType;

	ScalarImageReaderType::Pointer wmpvReader = ScalarImageReaderType::New();
	wmpvReader->SetFileName(wmpvImageFileName);
	wmpvReader->Update();

	ScalarImageReaderType::Pointer gmwmpvReader = ScalarImageReaderType::New();
	gmwmpvReader->SetFileName(gmwmpvImageFileName);
	gmwmpvReader->Update();

	ScalarImageReaderType::Pointer thicknessReader = ScalarImageReaderType::New();
	thicknessReader->SetFileName(thicknessImageFileName);
	thicknessReader->Update();

	ScalarImageReaderType::Pointer boundaryReader = ScalarImageReaderType::New();
	boundaryReader->SetFileName(boundaryImageFileName);
	boundaryReader->Update();

	FilterType::Pointer filter = FilterType::New();
	filter->SetWhiteMatterPVMap(wmpvReader->GetOutput());
	filter->SetWhitePlusGreyMatterPVMap(gmwmpvReader->GetOutput());
	filter->SetThicknessPriorMap(thicknessReader->GetOutput());
	filter->SetGWI(boundaryReader->GetOutput());
	filter->SetMaxIterations(maxIterations);
	filter->SetM(integrationSteps);
	filter->SetLambda(lambda);
	filter->SetUpdateSigma(updateSigma);
	filter->SetDeformationSigma(deformationSigma);
	filter->SetEpsilon(epsilon);
	filter->SetAlpha(alpha);
	filter->Update();
  std::cerr << "Finishing filter" << std::endl;

	// The ITK unit test framework compares the output image with a baseline
	ScalarImageWriterType::Pointer writer = ScalarImageWriterType::New();
	writer->SetInput(filter->GetOutput());
	writer->SetFileName(outputImage);
	writer->Update();

	std::cerr << "Finishing main" << std::endl;

	return EXIT_SUCCESS;
}
