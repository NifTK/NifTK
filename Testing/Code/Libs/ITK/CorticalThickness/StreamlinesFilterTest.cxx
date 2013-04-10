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
#include "itkLaplacianSolverImageFilter.h"
#include "itkScalarImageToNormalizedGradientVectorImageFilter.h"
#include "itkIntegrateStreamlinesFilter.h"
#include "itkRelaxStreamlinesFilter.h"
#include "itkOrderedTraversalStreamlinesFilter.h"
#include "ConversionUtils.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCastImageFilter.h"

/**
 * Basic tests for IntegrateStreamlinesFilter
 */
int StreamlinesFilterTest(int argc, char * argv[])
{
  if (argc < 18 )
    {
    	std::cerr << "StreamlinesFilterTest inputImage outputImage gm wm csf lowVoltage highVoltage laplaceMaxIters laplaceEpsilon useOpt stepSize relaxMaxIters relaxEpsilon pixelX pixelY expectedValue tolerance" << std::endl;
    	return EXIT_FAILURE;
    }

  // Define the dimension of the images
  const unsigned int Dimension = 2;
  typedef float ScalarType;

  // Read args
  std::string inputImage = argv[1];
  std::string outputImage = argv[2];
  ScalarType gmLabel = niftk::ConvertToDouble(argv[3]);
  ScalarType wmLabel = niftk::ConvertToDouble(argv[4]);
  ScalarType csfLabel = niftk::ConvertToDouble(argv[5]);
  double lowVoltage = niftk::ConvertToDouble(argv[6]);
  double highVoltage = niftk::ConvertToDouble(argv[7]);
  int laplaceMaxIters = niftk::ConvertToInt(argv[8]);
  double laplaceEpsilon = niftk::ConvertToDouble(argv[9]);
  std::string useOpt = argv[10];
  double stepSize = niftk::ConvertToDouble(argv[11]);
  int relaxMaxIters = niftk::ConvertToInt(argv[12]);
  double relaxEpsilon = niftk::ConvertToDouble(argv[13]);
  int pixelX = niftk::ConvertToInt(argv[14]);
  int pixelY = niftk::ConvertToInt(argv[15]);
  double expectedValue = niftk::ConvertToDouble(argv[16]);
  double tolerance = niftk::ConvertToDouble(argv[17]);

  typedef itk::Image< ScalarType, Dimension >   ImageType;
  typedef itk::ImageFileReader< ImageType >     ReaderType;

  ReaderType::Pointer reader  = ReaderType::New();
  reader->SetFileName( inputImage );
  
  typedef itk::LaplacianSolverImageFilter<ImageType, ScalarType> LaplacianFilterType;
  LaplacianFilterType::Pointer laplaceFilter = LaplacianFilterType::New();
  laplaceFilter->SetInput(reader->GetOutput());
  laplaceFilter->SetLowVoltage(lowVoltage);
  laplaceFilter->SetHighVoltage(highVoltage);
  laplaceFilter->SetMaximumNumberOfIterations(laplaceMaxIters);
  laplaceFilter->SetEpsilonConvergenceThreshold(laplaceEpsilon);
  laplaceFilter->SetLabelThresholds(gmLabel, wmLabel, csfLabel); 
  
  if (useOpt == "ON")
    {
    	laplaceFilter->SetUseGaussSeidel(true);
    }
  else
    {
    	laplaceFilter->SetUseGaussSeidel(false);
    }
    
  typedef itk::ScalarImageToNormalizedGradientVectorImageFilter<ImageType, ScalarType> NormalsFilterType;
  NormalsFilterType::Pointer normalsFilter = NormalsFilterType::New();
  normalsFilter->SetInput(laplaceFilter->GetOutput());
 
  // First option is to integrate, as per Jones et al. 2000.
  typedef itk::IntegrateStreamlinesFilter< ImageType, ScalarType, Dimension > IntegrateFilterType;
  IntegrateFilterType::Pointer integrateFilter = IntegrateFilterType::New();
  integrateFilter->SetScalarImage(laplaceFilter->GetOutput());
  integrateFilter->SetVectorImage(normalsFilter->GetOutput());
  integrateFilter->SetMinIterationVoltage(lowVoltage);
  integrateFilter->SetMaxIterationVoltage(highVoltage);
  integrateFilter->SetStepSize(stepSize);
  integrateFilter->SetMaxIterationLength(200);
  
  // Or we can solve a PDE by relaxation, as per Yezzi and Prince 2003.
  typedef itk::RelaxStreamlinesFilter< ImageType, ScalarType, Dimension > RelaxFilterType;
  RelaxFilterType::Pointer relaxFilter = RelaxFilterType::New();
  relaxFilter->SetScalarImage(laplaceFilter->GetOutput());
  relaxFilter->SetVectorImage(normalsFilter->GetOutput());
  relaxFilter->SetSegmentedImage(reader->GetOutput());
  relaxFilter->SetInitializeBoundaries(false);
  relaxFilter->SetLowVoltage(lowVoltage);
  relaxFilter->SetHighVoltage(highVoltage);
  relaxFilter->SetMaximumNumberOfIterations(relaxMaxIters);
  relaxFilter->SetEpsilonConvergenceThreshold(relaxEpsilon);
  relaxFilter->SetLabelThresholds(gmLabel, wmLabel, csfLabel); 
  relaxFilter->SetMaximumLength(10000);
  
  // Or, we can solve PDE by ordered traversal, as per Yezzi and Prince 2003.
  typedef itk::OrderedTraversalStreamlinesFilter< ImageType, ScalarType, Dimension > OrderedTraversalFilterType;
  OrderedTraversalFilterType::Pointer traversalFilter = OrderedTraversalFilterType::New();
  traversalFilter->SetScalarImage(laplaceFilter->GetOutput());
  traversalFilter->SetVectorImage(normalsFilter->GetOutput());
  traversalFilter->SetLowVoltage(lowVoltage);
  traversalFilter->SetHighVoltage(highVoltage);
  
  // Set up an output image.
  typedef unsigned char OutputPixelType;  
  typedef itk::Image<OutputPixelType, Dimension>                OutputImageType; 
  typedef itk::CastImageFilter<ImageType, OutputImageType>      CastFilterType;
  typedef itk::ImageFileWriter< OutputImageType >               WriterType;
  
  CastFilterType::Pointer caster = CastFilterType::New();
  WriterType::Pointer writer = WriterType::New();

  typedef ImageType::IndexType IndexType;
  IndexType index;
  
  index[0] = pixelX;
  index[1] = pixelY;
  
  typedef ImageType::PixelType PixelType;
  PixelType pixel;
  
    
  if (stepSize > 0)
    {
      integrateFilter->Update();
      caster->SetInput(integrateFilter->GetOutput());
      pixel = integrateFilter->GetOutput()->GetPixel(index);	
    }
  else if (relaxMaxIters > 0)
    {
      relaxFilter->Update();
      caster->SetInput(relaxFilter->GetOutput());
      pixel = relaxFilter->GetOutput()->GetPixel(index);		
    }
  else
    {
      traversalFilter->Update();
      caster->SetInput(traversalFilter->GetOutput());
      pixel = traversalFilter->GetOutput()->GetPixel(index);	
    }
  
  writer->SetInput(caster->GetOutput());
  writer->SetFileName(outputImage);
  writer->Update();
  
  if (fabs(pixel - expectedValue) > tolerance) 
    {
    	std::cerr << "Expected value:" << expectedValue << ", but got:" << pixel << std::endl;
    	return EXIT_FAILURE;
    }
  
  // We are done. Go for coffee.
  return EXIT_SUCCESS;    
}
