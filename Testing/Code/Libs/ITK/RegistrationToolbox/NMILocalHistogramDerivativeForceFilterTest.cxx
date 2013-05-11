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
#include <ConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkTranslationTransform.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkNMIImageToImageMetric.h>
#include <itkNMILocalHistogramDerivativeForceFilter.h>

int NMILocalHistogramDerivativeForceFilterTest(int argc, char * argv[])
{
  if( argc < 9)
    {
    std::cerr << "Usage   : NMILocalHistogramDerivativeForceFilterTest img1 img2 expectedTotalNMI tolerance testPointX testPointY forceX forceY" << std::endl;
    return 1;
    }
  std::cerr << "img1:" << argv[1] << std::endl;
  std::cerr << "img2:" << argv[2] << std::endl;
  std::cerr << "expectedNMI:" << argv[3] << std::endl;
  std::cerr << "tolerance:" << argv[4] << std::endl;
  std::cerr << "testPointX:" << argv[5] << std::endl;
  std::cerr << "testPointY:" << argv[6] << std::endl;
  std::cerr << "forceX:" << argv[7] << std::endl;
  std::cerr << "forceY:" << argv[8] << std::endl;
  
  const     unsigned int   Dimension = 2;
  typedef   unsigned char  PixelType;

  typedef itk::Image< PixelType, Dimension >      ImageType;
  typedef itk::Vector<double, Dimension>          VectorPixelType;
  typedef itk::Image<VectorPixelType, Dimension > VectorImageType;
  typedef itk::ImageFileReader< ImageType >       ReaderType;
  typedef VectorImageType::IndexType              VectorIndexType;
  typedef VectorImageType::SizeType               VectorSizeType;
  
  ReaderType::Pointer fixedReader  = ReaderType::New();
  ReaderType::Pointer movingReader = ReaderType::New();

  fixedReader->SetFileName(  argv[1] );
  movingReader->SetFileName( argv[2] );

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

  typedef itk::NearestNeighborInterpolateImageFunction<ImageType, double >  InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  typedef itk::TranslationTransform< double, Dimension >  TransformType;
  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  
  typedef itk::NMIImageToImageMetric<ImageType, ImageType> MetricType;
  MetricType::Pointer metric = MetricType::New();
  
  typedef itk::NMILocalHistogramDerivativeForceFilter<ImageType, ImageType, float> ForceFilterType;
  ForceFilterType::Pointer filter = ForceFilterType::New();
  
  MetricType::TransformParametersType displacement( Dimension );
  displacement[0] = 0;
  displacement[1] = 0;
  
  metric->SetFixedImage(fixedImage);
  metric->SetMovingImage(movingImage);
  metric->SetTransform(transform);
  metric->SetInterpolator(interpolator);
  metric->SetHistogramSize(5, 5);
  metric->SetIntensityBounds(127,250,1,250);
  metric->Initialize();
  double nmi = metric->GetValue(displacement);
  
  double expectedNMI = niftk::ConvertToDouble(argv[3]);
  double tolerance = niftk::ConvertToDouble(argv[4]);
  long int testX = niftk::ConvertToInt(argv[5]);
  long int testY = niftk::ConvertToInt(argv[6]);
  double forceX = niftk::ConvertToDouble(argv[7]);
  double forceY = niftk::ConvertToDouble(argv[8]);
  
  if (fabs(nmi - expectedNMI) > tolerance) return EXIT_FAILURE;
  
  filter->SetInput(0, fixedImage);
  filter->SetInput(1, movingImage);
  filter->SetMetric(metric);
  filter->Update();
  
  VectorSizeType size;
  VectorIndexType index;
  VectorPixelType pixel;
  
  // Dump the image to log file.
  size = filter->GetOutput()->GetLargestPossibleRegion().GetSize();
  for (unsigned int x = 0; x < size[0]; x++)
    {
      for (unsigned int y = 0; y < size[1]; y++)
        {
          index[0] = x;
          index[1] = y;
          std::cerr << "[" << x << "," << y << "] = " << filter->GetOutput()->GetPixel(index) << std::endl;
        }
    }

  // Need to actually check force.
  index[0] = testX;
  index[1] = testY;
  pixel = filter->GetOutput()->GetPixel(index);
  
  if (fabs(pixel[0] - forceX) > tolerance) return EXIT_FAILURE;
  if (fabs(pixel[1] - forceY) > tolerance) return EXIT_FAILURE;
  
  return EXIT_SUCCESS;
}
