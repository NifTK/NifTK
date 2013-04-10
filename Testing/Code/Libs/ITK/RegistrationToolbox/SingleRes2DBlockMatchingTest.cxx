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
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegistrationFactory.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"
#include "itkBlockMatchingMethod.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"
#include "itkPowellOptimizer.h"
#include "itkUCLSimplexOptimizer.h"
#include "itkSimilarityMeasure.h"
#include "itkTransform.h"
#include "itkAbsoluteManhattanDistancePointMetric.h"

/**
 * SingleRes2DBlocMatchingTest.
 * 
 */
int SingleRes2DBlockMatchingTest(int argc, char * argv[])
{
  if( argc < 12)
    {
      std::cerr << "Usage   : SingleRes2DBlockMatchingTest fixedImg movingImg outputImg tx ty rz exTx exTy exRz exIters tolerance" << std::endl;
      return 1;
    }

  const unsigned int Dimension = 2;
  typedef unsigned char InputPixelType;
  typedef unsigned char OutputPixelType;
  typedef double DataType;
  
  // Parse Input 
  std::string fixedImage = argv[1];
  std::string movingImage = argv[2];
  std::string resampledImage = argv[3];
  double tx = niftk::ConvertToDouble(argv[4]);
  double ty = niftk::ConvertToDouble(argv[5]);
  double rz = niftk::ConvertToDouble(argv[6]);
  double exTx = niftk::ConvertToDouble(argv[7]);
  double exTy = niftk::ConvertToDouble(argv[8]);
  double exRz = niftk::ConvertToDouble(argv[9]);
  double tolerance = niftk::ConvertToDouble(argv[10]);
  
  // Load images.
  typedef itk::Image< InputPixelType, Dimension> InputImageType;
  typedef itk::ImageFileReader< InputImageType  >  ImageReaderType;

  ImageReaderType::Pointer fixedImageReader  = ImageReaderType::New();
  ImageReaderType::Pointer movingImageReader = ImageReaderType::New();

  fixedImageReader->SetFileName(  fixedImage );
  fixedImageReader->Update();
  
  movingImageReader->SetFileName( movingImage );
  movingImageReader->Update();
  
  typedef itk::ImageRegistrationFactory<InputImageType,Dimension, double> FactoryType;
  typedef itk::SingleResolutionImageRegistrationBuilder<InputImageType, Dimension, double> BuilderType;
  typedef itk::MaskedImageRegistrationMethod<InputImageType> ImageRegistrationMethodType;
  typedef itk::SimilarityMeasure<InputImageType, InputImageType> SimilarityMeasureType;
  typedef itk::BlockMatchingMethod<InputImageType, DataType> BlockMatchingType;
  typedef BlockMatchingType* BlockMatchingPointer;
  typedef BlockMatchingType::PointSetType PointSetType;
  typedef itk::AbsoluteManhattanDistancePointMetric<PointSetType, PointSetType> PointSetMetricType;  
  
  // Start building
  BuilderType::Pointer builder = BuilderType::New();
  builder->StartCreation((itk::SingleResRegistrationMethodTypeEnum)5);                    // Block matching
  builder->CreateInterpolator((itk::InterpolationTypeEnum)4);                             // Sinc
  SimilarityMeasureType::Pointer metric = builder->CreateMetric((itk::MetricTypeEnum)1);  // SSD
  builder->CreateTransform((itk::TransformTypeEnum)2, fixedImageReader->GetOutput());     // Rigid
  builder->CreateOptimizer((itk::OptimizerTypeEnum)5);     // Powell
  ImageRegistrationMethodType::Pointer method = builder->GetSingleResolutionImageRegistrationMethod();
  BlockMatchingType::Pointer blockMatchingMethod = static_cast<BlockMatchingPointer>(method.GetPointer());
  PointSetMetricType::Pointer pointMetric = PointSetMetricType::New();
  
  typedef ImageRegistrationMethodType::ParametersType ParametersType;
  ParametersType initialParameters(method->GetTransform()->GetNumberOfParameters());
  initialParameters.Fill(0);
  initialParameters[0] = tx;
  initialParameters[1] = ty;
  initialParameters[2] = rz;

  metric->SetIntensityBounds(0, std::numeric_limits<InputPixelType>::max(), 0, std::numeric_limits<InputPixelType>::max());
  metric->SetWeightingFactor(0);  
  metric->SetPrintOutMetricEvaluation(false);
  
  blockMatchingMethod->SetPointSetMetric(pointMetric);
  blockMatchingMethod->SetWriteTransformedMovingImage(false);
  blockMatchingMethod->SetFixedImage(fixedImageReader->GetOutput());
  blockMatchingMethod->SetMovingImage(movingImageReader->GetOutput());
  blockMatchingMethod->SetInitialTransformParameters(initialParameters);
  blockMatchingMethod->SetBlockParameters(22, 22, 4, 2);
  blockMatchingMethod->SetMinimumBlockSize(12);
  blockMatchingMethod->SetEpsilon(10);
  blockMatchingMethod->Update();

  if (fabs(exTx - method->GetLastTransformParameters()[0]) > tolerance) return EXIT_FAILURE;
  if (fabs(exTy - method->GetLastTransformParameters()[1]) > tolerance) return EXIT_FAILURE;
  if (fabs(exRz - method->GetLastTransformParameters()[2]) > tolerance) return EXIT_FAILURE;
  
  return EXIT_SUCCESS;    
}
