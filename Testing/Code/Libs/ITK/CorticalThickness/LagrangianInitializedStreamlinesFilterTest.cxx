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
#include "itkImage.h"
#include "itkCheckForThreeLevelsFilter.h"
#include "itkLaplacianSolverImageFilter.h"
#include "itkScalarImageToNormalizedGradientVectorImageFilter.h"
#include "itkLagrangianInitializedRelaxStreamlinesFilter.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCastImageFilter.h"

int LagrangianInitializedStreamlinesFilterTest(int argc, char * argv[])
{

  if (argc < 2 )
    {
      std::cerr << "LagrangianInitializedStreamlinesFilterTest outputImage" << std::endl;
      return EXIT_FAILURE;
    }

  std::string outputImage = argv[1];
  
  // Define the dimension of the images
  const unsigned int Dimension = 2;
  typedef float ScalarType;
  typedef unsigned char OutputPixelType;  
  typedef itk::Image< ScalarType, Dimension >   ImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;  
  typedef itk::ImageRegion<Dimension>           RegionType;
  typedef ImageType::SizeType                   SizeType;
  typedef ImageType::IndexType                  IndexType;
  
  // Create a test input image.
  // 0 0 0 0 0 0 0 0
  // 0 1 1 1 0 1 1 0
  // 0 1 1 1 1 1 1 0
  // 0 1 1 1 1 1 1 0
  // 0 1 1 2 2 1 1 0
  // 0 1 1 1 1 1 1 0
  // 0 1 1 1 1 1 1 0
  // 0 0 0 0 0 0 0 0
  ImageType::Pointer image = ImageType::New();
  SizeType size;
  size[0] = 8;
  size[1] = 8;
  IndexType index;
  index[0] = 0;
  index[1] = 0;
  RegionType region;
  region.SetSize(size);
  region.SetIndex(index);
  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(0);
  index[0] = 1; index[1] = 1; image->SetPixel(index, 1);
  index[0] = 2; index[1] = 1; image->SetPixel(index, 1);
  index[0] = 3; index[1] = 1; image->SetPixel(index, 1);
  index[0] = 5; index[1] = 1; image->SetPixel(index, 1);
  index[0] = 6; index[1] = 1; image->SetPixel(index, 1);
  index[0] = 1; index[1] = 2; image->SetPixel(index, 1);
  index[0] = 2; index[1] = 2; image->SetPixel(index, 1);
  index[0] = 3; index[1] = 2; image->SetPixel(index, 1);
  index[0] = 4; index[1] = 2; image->SetPixel(index, 1);
  index[0] = 5; index[1] = 2; image->SetPixel(index, 1);
  index[0] = 6; index[1] = 2; image->SetPixel(index, 1);
  index[0] = 1; index[1] = 3; image->SetPixel(index, 1);
  index[0] = 2; index[1] = 3; image->SetPixel(index, 1);
  index[0] = 3; index[1] = 3; image->SetPixel(index, 1);
  index[0] = 4; index[1] = 3; image->SetPixel(index, 1);
  index[0] = 5; index[1] = 3; image->SetPixel(index, 1);
  index[0] = 6; index[1] = 3; image->SetPixel(index, 1);
  index[0] = 1; index[1] = 4; image->SetPixel(index, 1);
  index[0] = 2; index[1] = 4; image->SetPixel(index, 1);
  index[0] = 3; index[1] = 4; image->SetPixel(index, 2);
  index[0] = 4; index[1] = 4; image->SetPixel(index, 2);
  index[0] = 5; index[1] = 4; image->SetPixel(index, 1);
  index[0] = 6; index[1] = 4; image->SetPixel(index, 1);
  index[0] = 1; index[1] = 5; image->SetPixel(index, 1);
  index[0] = 2; index[1] = 5; image->SetPixel(index, 1);
  index[0] = 3; index[1] = 5; image->SetPixel(index, 1);
  index[0] = 4; index[1] = 5; image->SetPixel(index, 1);
  index[0] = 5; index[1] = 5; image->SetPixel(index, 1);
  index[0] = 6; index[1] = 5; image->SetPixel(index, 1);
  index[0] = 1; index[1] = 6; image->SetPixel(index, 1);
  index[0] = 2; index[1] = 6; image->SetPixel(index, 1);
  index[0] = 3; index[1] = 6; image->SetPixel(index, 1);
  index[0] = 4; index[1] = 6; image->SetPixel(index, 1);
  index[0] = 5; index[1] = 6; image->SetPixel(index, 1);
  index[0] = 6; index[1] = 6; image->SetPixel(index, 1);

  // Similarly create a PV map.
  ImageType::Pointer pvMap = ImageType::New();
  pvMap->SetRegions(region);
  pvMap->Allocate();
  pvMap->FillBuffer(0.3);
  index[0] = 1; index[1] = 1; pvMap->SetPixel(index, 1);
  index[0] = 2; index[1] = 1; pvMap->SetPixel(index, 1);
  index[0] = 3; index[1] = 1; pvMap->SetPixel(index, 1);
  index[0] = 5; index[1] = 1; pvMap->SetPixel(index, 1);
  index[0] = 6; index[1] = 1; pvMap->SetPixel(index, 1);
  index[0] = 1; index[1] = 2; pvMap->SetPixel(index, 1);
  index[0] = 2; index[1] = 2; pvMap->SetPixel(index, 1);
  index[0] = 3; index[1] = 2; pvMap->SetPixel(index, 1);
  index[0] = 4; index[1] = 2; pvMap->SetPixel(index, 0.3);
  index[0] = 5; index[1] = 2; pvMap->SetPixel(index, 1);
  index[0] = 6; index[1] = 2; pvMap->SetPixel(index, 1);
  index[0] = 1; index[1] = 3; pvMap->SetPixel(index, 1);
  index[0] = 2; index[1] = 3; pvMap->SetPixel(index, 1);
  index[0] = 3; index[1] = 3; pvMap->SetPixel(index, 1);
  index[0] = 4; index[1] = 3; pvMap->SetPixel(index, 1);
  index[0] = 5; index[1] = 3; pvMap->SetPixel(index, 1);
  index[0] = 6; index[1] = 3; pvMap->SetPixel(index, 1);
  index[0] = 1; index[1] = 4; pvMap->SetPixel(index, 1);
  index[0] = 2; index[1] = 4; pvMap->SetPixel(index, 1);
  index[0] = 3; index[1] = 4; pvMap->SetPixel(index, 0.3);
  index[0] = 4; index[1] = 4; pvMap->SetPixel(index, 0.4);
  index[0] = 5; index[1] = 4; pvMap->SetPixel(index, 1);
  index[0] = 6; index[1] = 4; pvMap->SetPixel(index, 1);
  index[0] = 1; index[1] = 5; pvMap->SetPixel(index, 1);
  index[0] = 2; index[1] = 5; pvMap->SetPixel(index, 1);
  index[0] = 3; index[1] = 5; pvMap->SetPixel(index, 1);
  index[0] = 4; index[1] = 5; pvMap->SetPixel(index, 1);
  index[0] = 5; index[1] = 5; pvMap->SetPixel(index, 1);
  index[0] = 6; index[1] = 5; pvMap->SetPixel(index, 1);
  index[0] = 1; index[1] = 6; pvMap->SetPixel(index, 1);
  index[0] = 2; index[1] = 6; pvMap->SetPixel(index, 1);
  index[0] = 3; index[1] = 6; pvMap->SetPixel(index, 1);
  index[0] = 4; index[1] = 6; pvMap->SetPixel(index, 1);
  index[0] = 5; index[1] = 6; pvMap->SetPixel(index, 1);
  index[0] = 6; index[1] = 6; pvMap->SetPixel(index, 1);
  
  typedef itk::CheckForThreeLevelsFilter<ImageType> CheckFilterType;
  CheckFilterType::Pointer checkFilter = CheckFilterType::New();
  checkFilter->SetSegmentedImage(image);
  checkFilter->SetLabelThresholds(1, 2, 0);
  checkFilter->Update();
  
  typedef itk::LaplacianSolverImageFilter<ImageType> LaplacianFilterType;
  LaplacianFilterType::Pointer laplaceFilter = LaplacianFilterType::New();
  laplaceFilter->SetInput(checkFilter->GetOutput());
  laplaceFilter->SetLowVoltage(0);
  laplaceFilter->SetHighVoltage(10000);
  laplaceFilter->SetMaximumNumberOfIterations(200);
  laplaceFilter->SetEpsilonConvergenceThreshold(0.00001);
  laplaceFilter->SetLabelThresholds(1, 2, 0); 
  laplaceFilter->SetUseGaussSeidel(true);
  laplaceFilter->Update();
  
  typedef itk::ScalarImageToNormalizedGradientVectorImageFilter<ImageType, ScalarType> NormalsFilterType;
  NormalsFilterType::Pointer normalsFilter = NormalsFilterType::New();
  normalsFilter->SetInput(laplaceFilter->GetOutput());
  normalsFilter->SetNormalize(true);
  normalsFilter->SetDerivativeType(NormalsFilterType::DERIVATIVE_OF_GAUSSIAN);
  normalsFilter->Update();
  
  typedef itk::LagrangianInitializedRelaxStreamlinesFilter< ImageType, ScalarType, Dimension > RelaxFilterType;
  RelaxFilterType::Pointer relaxFilter = RelaxFilterType::New();
  relaxFilter->SetSegmentedImage(image);
  relaxFilter->SetGMPVMap(pvMap);
  relaxFilter->SetScalarImage(laplaceFilter->GetOutput());
  relaxFilter->SetVectorImage(normalsFilter->GetOutput());
  relaxFilter->SetLabelThresholds(1, 2, 0); 
  relaxFilter->SetLowVoltage(0);
  relaxFilter->SetHighVoltage(10000);
  relaxFilter->SetMaximumNumberOfIterations(200);
  relaxFilter->SetEpsilonConvergenceThreshold(0.00001);
  relaxFilter->SetMaximumSearchDistance(100);
  relaxFilter->Update();

  // Set up an output image.
  typedef itk::RescaleIntensityImageFilter<ImageType, ImageType > RescalerType;  
  RescalerType::Pointer outputRescaler = RescalerType::New();
  outputRescaler->SetInput(relaxFilter->GetOutput());
  outputRescaler->SetOutputMinimum( 0 );
  outputRescaler->SetOutputMaximum( 255 );

  typedef itk::CastImageFilter<ImageType, OutputImageType> CastFilterType;    
  CastFilterType::Pointer caster = CastFilterType::New();
  caster->SetInput(outputRescaler->GetOutput());
  
  typedef itk::ImageFileWriter< OutputImageType > WriterType;  
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(caster->GetOutput());
  writer->SetFileName(outputImage);
  writer->Update();
  
  
  // Actually check something, or else its not a Unit Test!!!
  double result;
  double tolerance = 0.00001;
  index[0] = 1; index[1] = 1; 
  result = relaxFilter->GetOutput()->GetPixel(index);
  if (fabs(result - 4.51743) > tolerance) {
    std::cout << "Expected 4.51743, but got " << result << std::endl;
    return EXIT_FAILURE;
  }

  index[0] = 3; index[1] = 3; 
  result = relaxFilter->GetOutput()->GetPixel(index);
  if (fabs(result - 3.2991) > tolerance) {
    std::cout << "Expected 3.2991, but got " << result << std::endl;
    return EXIT_FAILURE;
  }
  
  index[0] = 2; index[1] = 2;
  result = relaxFilter->GetOutput()->GetPixel(index); 
  if (fabs(result - 3.87331) > tolerance) {
    std::cout << "Expected 3.87331, but got " << result << std::endl;
    return EXIT_FAILURE;
  }
  
  // We are done. Go for coffee.
  return EXIT_SUCCESS;    
}
