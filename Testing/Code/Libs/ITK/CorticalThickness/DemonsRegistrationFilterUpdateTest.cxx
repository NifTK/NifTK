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

#include <itkDemonsRegistrationFilter.h>
#include <itkImage.h>
#include <itkVector.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

/**
 * Test the DemonsRegistrationFilter.
 */
int DemonsRegistrationFilterUpdateTest(int argc, char* argv[])
{
  if (argc != 4)
  {
    std::cerr << "Usage: DemonsRegistrationFilterUpdateTest wmpvImage gmwmpvImage outputImage" << std::endl;
    return EXIT_FAILURE;
  }
  std::string wmpvImageFileName = argv[1];
  std::string gmwmpvImageFileName = argv[2];
  std::string outputImageFileName = argv[3];

  typedef float ScalarType;
  typedef itk::Vector<ScalarType, 2> VectorType;
  typedef itk::Image<ScalarType, 2> ScalarImageType;
  typedef itk::Image<VectorType, 2> VectorImageType;
  typedef itk::ImageFileReader<ScalarImageType> ScalarImageReaderType;
  typedef itk::ImageFileWriter<VectorImageType> VectorImageWriterType;
  typedef itk::DemonsRegistrationFilter<ScalarImageType, ScalarImageType, VectorImageType> UpdateFilterType;

  ScalarImageReaderType::Pointer wmpvReader = ScalarImageReaderType::New();
  wmpvReader->SetFileName(wmpvImageFileName);
  wmpvReader->Update();

  ScalarImageReaderType::Pointer gmwmpvReader = ScalarImageReaderType::New();
  gmwmpvReader->SetFileName(gmwmpvImageFileName);
  gmwmpvReader->Update();

  VectorImageType::Pointer zeroDeformationField = VectorImageType::New();
  zeroDeformationField->SetRegions(wmpvReader->GetOutput()->GetLargestPossibleRegion());
  zeroDeformationField->SetSpacing(wmpvReader->GetOutput()->GetSpacing());
  zeroDeformationField->SetOrigin(wmpvReader->GetOutput()->GetOrigin());
  zeroDeformationField->SetDirection(wmpvReader->GetOutput()->GetDirection());
  zeroDeformationField->Allocate();

  VectorType zero;
  zero.Fill(0);

  zeroDeformationField->FillBuffer(zero);

  UpdateFilterType::Pointer updateFilter = UpdateFilterType::New();
  updateFilter->SetNumberOfIterations(1);
  updateFilter->SetFixedImage(gmwmpvReader->GetOutput());
  updateFilter->SetMovingImage(wmpvReader->GetOutput());
  updateFilter->SetInitialDeformationField(zeroDeformationField);
  updateFilter->SetIntensityDifferenceThreshold(0.001);
  updateFilter->SetUseImageSpacing(true);
  updateFilter->SetSmoothUpdateField(true);
  updateFilter->SetUpdateFieldStandardDeviations(1);
  updateFilter->SetSmoothDeformationField(false);
  updateFilter->SetUseMovingImageGradient(true);
  updateFilter->Modified();
  updateFilter->UpdateLargestPossibleRegion();

  VectorImageType::SizeType size;
  size = updateFilter->GetOutput()->GetLargestPossibleRegion().GetSize();

  VectorImageType::IndexType index;

  for (unsigned int y = 0; y < size[1]; y++)
  {
	  for (unsigned int x = 0; x < size[0]; x++)
	  {
		  index[0] = x;
		  index[1] = y;

		  std::cout << "At i=" << index << ", v=" << updateFilter->GetOutput()->GetPixel(index) << std::endl;
	  }
  }

  // The ITK unit test framework compares the output image with a baseline
  VectorImageWriterType::Pointer writer = VectorImageWriterType::New();
  writer->SetInput(updateFilter->GetOutput());
  writer->SetFileName(outputImageFileName);
  writer->Update();

  return EXIT_SUCCESS;
}
