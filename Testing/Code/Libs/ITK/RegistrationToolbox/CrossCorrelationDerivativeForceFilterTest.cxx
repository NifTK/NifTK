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
#include <niftkConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkTranslationTransform.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkNMIImageToImageMetric.h>
#include <itkCrossCorrelationDerivativeForceFilter.h>


int CrossCorrelationDerivativeForceFilterTest(int argc, char * argv[])
{
  if (argc < 10)
  {
    std::cerr << "Usage : CrossCorrelationDerivativeForceFilterTest movingImg fixedImage tolerance testPointX testPointY forceX forceY" << std::endl;
    return 1;
  }
  
  const unsigned int Dimension = 3;
  typedef unsigned char PixelType;

  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::Vector<double, Dimension> VectorPixelType;
  typedef itk::Image<VectorPixelType, Dimension> VectorImageType;
  typedef itk::ImageFileReader< ImageType> ReaderType;
  typedef VectorImageType::IndexType VectorIndexType;
  typedef VectorImageType::SizeType VectorSizeType;
  
  ReaderType::Pointer fixedReader = ReaderType::New();
  ReaderType::Pointer movingReader = ReaderType::New();

  movingReader->SetFileName(argv[1]);
  fixedReader->SetFileName(argv[2]);

  try
  {
    fixedReader->Update();
    movingReader->Update();
  }
  catch(itk::ExceptionObject& excep)
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }
  
  ImageType::ConstPointer fixedImage = fixedReader->GetOutput();
  ImageType::ConstPointer movingImage = movingReader->GetOutput();

  typedef itk::NearestNeighborInterpolateImageFunction<ImageType, double> InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  typedef itk::TranslationTransform<double, Dimension> TransformType;
  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  
  typedef itk::NMIImageToImageMetric<ImageType, ImageType> MetricType;
  MetricType::Pointer metric = MetricType::New();
  
  typedef itk::CrossCorrelationDerivativeForceFilter<ImageType, ImageType, double> ForceFilterType;
  ForceFilterType::Pointer filter = ForceFilterType::New();
  
  MetricType::TransformParametersType displacement(Dimension);
  displacement[0] = 0;
  displacement[1] = 0;
  displacement[2] = 0;
  
  metric->SetFixedImage(fixedImage);
  metric->SetMovingImage(movingImage);
  metric->SetTransform(transform);
  metric->SetInterpolator(interpolator);
  metric->Initialize();
  
  double tolerance = niftk::ConvertToDouble(argv[3]);
  long int testX = niftk::ConvertToInt(argv[4]);
  long int testY = niftk::ConvertToInt(argv[5]);
  long int testZ = niftk::ConvertToInt(argv[6]);
  double forceX = niftk::ConvertToDouble(argv[7]);
  double forceY = niftk::ConvertToDouble(argv[8]);
  double forceZ = niftk::ConvertToDouble(argv[9]);
  
  filter->SetInput(0, fixedImage);
  filter->SetInput(1, movingImage);
  filter->SetMetric(metric);
  filter->Update();
  
  VectorIndexType index;
  VectorPixelType pixel;
#if 0
  VectorSizeType size;
  
  // Dump the image to log file.
  size = filter->GetOutput()->GetLargestPossibleRegion().GetSize();
  for (unsigned int z = 0; z < size[2]; z++)
  {
    for (unsigned int y = 0; y < size[1]; y++)
    {
      for (unsigned int x = 0; x < size[0]; x++)
      {
        index[0] = x;
        index[1] = y;
        index[2] = z;
        std::cerr << "[" << x << "," << y << "," << z << "] = " << filter->GetOutput()->GetPixel(index) << std::endl;
      }
    }
  }
#endif
                           
  // Need to actually check force.
  index[0] = testX;
  index[1] = testY;
  index[2] = testZ;
  pixel = filter->GetOutput()->GetPixel(index);
  
  if (fabs(pixel[0] - forceX) > tolerance) 
    return EXIT_FAILURE;
  if (fabs(pixel[1] - forceY) > tolerance) 
    return EXIT_FAILURE;
  if (fabs(pixel[2] - forceZ) > tolerance) 
    return EXIT_FAILURE;
  
  std::cout << "Passed" << std::endl; 
  return EXIT_SUCCESS;
}  



