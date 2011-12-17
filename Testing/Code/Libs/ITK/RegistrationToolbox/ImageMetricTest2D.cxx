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
#include <iostream>
#include <memory>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTranslationTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageRegistrationFactory.h"
#include "itkSimilarityMeasure.h"
#include "ConversionUtils.h"

int ImageMetricTest2D(int argc, char * argv[])
{
  if( argc < 12)
    {
    std::cerr << "Usage   : ImageMetricTest2D metric img1 img2 tx ty fixedLow fixedHigh movingLow movingHigh expected samples" << std::endl;
    return 1;
    }
  std::cerr << "Metric:" << argv[1] << std::endl;
  std::cerr << "Image1:" << argv[2] << std::endl;
  std::cerr << "Image2:" << argv[3] << std::endl;
  std::cerr << "tx:" << argv[4] << std::endl;
  std::cerr << "ty:" << argv[5] << std::endl;
  std::cerr << "fixedLow:" << argv[6] << ":" << std::endl;
  std::cerr << "fixedHigh:" << argv[7] << ":" << std::endl;
  std::cerr << "movingLow:" << argv[8] << ":" << std::endl;
  std::cerr << "movingHigh:" << argv[9] << ":" << std::endl;
  std::cerr << "expect:" << argv[10] << std::endl;
  std::cerr << "samples:" << argv[11] << std::endl;
  
  const     unsigned int   Dimension = 2;
  typedef   float          PixelType;

  typedef itk::Image< PixelType, Dimension >   ImageType;
  typedef itk::ImageFileReader< ImageType >  ReaderType;

  ReaderType::Pointer fixedReader  = ReaderType::New();
  ReaderType::Pointer movingReader = ReaderType::New();

  fixedReader->SetFileName(  argv[2] );
  movingReader->SetFileName( argv[3] );

  try
    {
      fixedReader->Update();
      movingReader->Update();
    }
  catch( itk::ExceptionObject & excep )
    {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
    }

  ImageType::ConstPointer fixedImage  = fixedReader->GetOutput();
  ImageType::ConstPointer movingImage = movingReader->GetOutput();

  typedef itk::TranslationTransform< double, Dimension >  TransformType;
  TransformType::Pointer transform = TransformType::New();

  typedef itk::NearestNeighborInterpolateImageFunction<ImageType, double >  InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  transform->SetIdentity();

  typedef itk::ImageRegistrationFactory<ImageType, Dimension, double> ImageRegistrationFactoryType;
  ImageRegistrationFactoryType::Pointer factory = ImageRegistrationFactoryType::New();
  
  typedef itk::SimilarityMeasure<ImageType, ImageType > MetricType;
  typedef MetricType* SimilarityPointer;
  MetricType::Pointer metric = factory->CreateMetric((itk::MetricTypeEnum)niftk::ConvertToInt(argv[1]));
  
  metric->SetTransform(transform);
  metric->SetInterpolator(interpolator);
  metric->SetFixedImage(fixedImage);
  metric->SetMovingImage(movingImage);
  
  SimilarityPointer similarity = dynamic_cast<SimilarityPointer>(metric.GetPointer());
  similarity->SetIntensityBounds(
      (PixelType)niftk::ConvertToInt(argv[6]),
      (PixelType)niftk::ConvertToInt(argv[7]),
      (PixelType)niftk::ConvertToInt(argv[8]),
      (PixelType)niftk::ConvertToInt(argv[9]));
  
  try
    {
      metric->Initialize();
    }
  catch( itk::ExceptionObject & excep )
    {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
    }

  MetricType::TransformParametersType displacement( Dimension );
  displacement[0] = niftk::ConvertToDouble(argv[4]);
  displacement[1] = niftk::ConvertToDouble(argv[5]);
  
  const double value = similarity->GetValue( displacement );
  std::cout << displacement[0] << "   "  << displacement[1] << "   " << value << std::endl;
  
  double expected = niftk::ConvertToDouble(argv[10]);
  int samples = niftk::ConvertToInt(argv[11]);
  
  if (fabs(value - expected) > 0.00001) return EXIT_FAILURE;
  if (samples != similarity->GetNumberOfFixedSamples()) return EXIT_FAILURE;
  
  return EXIT_SUCCESS;
}
