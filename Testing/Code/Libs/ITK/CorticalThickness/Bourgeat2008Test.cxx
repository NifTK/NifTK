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
#include "itkCorrectGMUsingPVMapFilter.h"
#include "itkLaplacianSolverImageFilter.h"
#include "itkScalarImageToNormalizedGradientVectorImageFilter.h"
#include "itkLagrangianInitializedRelaxStreamlinesFilter.h"
#include "itkCastImageFilter.h"
#include "ConversionUtils.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCastImageFilter.h"
/**
 * Basic tests for Bourgeat et. al. 2008 paper.
 */
int Bourgeat2008Test(int argc, char * argv[])
{
  if (argc < 19 )
    {
    	std::cerr << "Bourgeat2008Test inputImage inputPVImage outputImage gm wm csf laplaceMaxIters laplaceEpsilon relaxMaxIters relaxEpsilon segThreshold rayThreshold minStep maxDist pixelX pixelY expectedValue tolerance" << std::endl;
    	return EXIT_FAILURE;
    }

  // Define the dimension of the images
  const unsigned int Dimension = 2;
  typedef double ScalarType;
  typedef unsigned char OutputPixelType;  
  typedef itk::Image<ScalarType, Dimension > ImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType; 
  typedef itk::ImageFileReader< ImageType > ReaderType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType;
  typedef itk::CastImageFilter<ImageType, OutputImageType> CastFilterType;  
  typedef itk::CorrectGMUsingPVMapFilter<ImageType> CorrectGMFilter;
  typedef itk::LaplacianSolverImageFilter<ImageType, ScalarType> LaplaceFilterType;
  typedef itk::ScalarImageToNormalizedGradientVectorImageFilter<ImageType, ScalarType> NormalsFilterType;
  typedef itk::LagrangianInitializedRelaxStreamlinesFilter<ImageType, ScalarType, Dimension> RelaxFilterType;
  typedef itk::RescaleIntensityImageFilter<ImageType, ImageType > RescalerType;
  
  // Read args
  std::string inputImage = argv[1];
  std::string inputPVImage = argv[2];
  std::string outputImage = argv[3];
  ScalarType gmLabel = niftk::ConvertToDouble(argv[4]);
  ScalarType wmLabel = niftk::ConvertToDouble(argv[5]);
  ScalarType csfLabel = niftk::ConvertToDouble(argv[6]);
  int laplaceMaxIters = niftk::ConvertToInt(argv[7]);
  double laplaceEpsilon = niftk::ConvertToDouble(argv[8]);
  int relaxMaxIters = niftk::ConvertToInt(argv[9]);
  double relaxEpsilon = niftk::ConvertToDouble(argv[10]);
  double segThreshold = niftk::ConvertToDouble(argv[11]);
  double rayThreshold = niftk::ConvertToDouble(argv[12]);
  double minStep = niftk::ConvertToDouble(argv[13]);
  double maxDist = niftk::ConvertToDouble(argv[14]);
  int pixelX = niftk::ConvertToInt(argv[15]);
  int pixelY = niftk::ConvertToInt(argv[16]);
  double expectedValue = niftk::ConvertToDouble(argv[17]);
  double tolerance = niftk::ConvertToDouble(argv[18]);
  double low = 0;
  double high = 10000;
  
  ReaderType::Pointer imageReader  = ReaderType::New();
  imageReader->SetFileName( inputImage );
  imageReader->Update();
  
  ReaderType::Pointer pvReader  = ReaderType::New();
  pvReader->SetFileName( inputPVImage );
  pvReader->Update();
  
  // Need to rescale PV map, as its meant to be 0-1.
  RescalerType::Pointer rescaler = RescalerType::New();
  rescaler->SetOutputMinimum( 0 );
  rescaler->SetOutputMaximum( 1 );
  rescaler->SetInput(pvReader->GetOutput());
  rescaler->Update();
  
  CorrectGMFilter::Pointer correctGMFilter = CorrectGMFilter::New();
  correctGMFilter->SetLabelThresholds(gmLabel, wmLabel, csfLabel); 
  correctGMFilter->SetSegmentedImage(imageReader->GetOutput());
  correctGMFilter->SetGMPVMap(rescaler->GetOutput());
  correctGMFilter->SetGreyMatterThreshold(segThreshold);
  correctGMFilter->Update();

  LaplaceFilterType::Pointer laplaceFilter = LaplaceFilterType::New();
  laplaceFilter->SetSegmentedImage(correctGMFilter->GetOutput());
  laplaceFilter->SetLowVoltage(low);
  laplaceFilter->SetHighVoltage(high);
  laplaceFilter->SetMaximumNumberOfIterations(laplaceMaxIters);
  laplaceFilter->SetEpsilonConvergenceThreshold(laplaceEpsilon);
  laplaceFilter->SetLabelThresholds(gmLabel, wmLabel, csfLabel); 
  laplaceFilter->SetUseGaussSeidel(true);
  laplaceFilter->Update();
  
  NormalsFilterType::Pointer normalsFilter = NormalsFilterType::New();
  normalsFilter->SetScalarImage(laplaceFilter->GetOutput());
  normalsFilter->Update();
  
  RelaxFilterType::Pointer relaxFilter = RelaxFilterType::New();
  relaxFilter->SetScalarImage(laplaceFilter->GetOutput());
  relaxFilter->SetVectorImage(normalsFilter->GetOutput());
  relaxFilter->SetSegmentedImage(correctGMFilter->GetOutput());
  relaxFilter->SetGMPVMap(rescaler->GetOutput());
  relaxFilter->SetLowVoltage(low);
  relaxFilter->SetHighVoltage(high);
  relaxFilter->SetMaximumNumberOfIterations(relaxMaxIters);
  relaxFilter->SetEpsilonConvergenceThreshold(relaxEpsilon); 
  relaxFilter->SetStepSizeThreshold(minStep);
  relaxFilter->SetMaximumSearchDistance(maxDist);
  relaxFilter->SetMaximumLength(10000);
  relaxFilter->SetGreyMatterPercentage(rayThreshold); 
  relaxFilter->SetLabelThresholds(gmLabel, wmLabel, csfLabel); 
  relaxFilter->Update();
  
  // Set up an output image.
  RescalerType::Pointer outputRescaler = RescalerType::New();
  outputRescaler->SetInput(relaxFilter->GetOutput());
  outputRescaler->SetOutputMinimum( 0 );
  outputRescaler->SetOutputMaximum( 255 );
  
  CastFilterType::Pointer caster = CastFilterType::New();
  caster->SetInput(outputRescaler->GetOutput());
  
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(caster->GetOutput());
  writer->SetFileName(outputImage);
  writer->Update();
  
  // Now check a value or two.
  
  typedef ImageType::IndexType IndexType;
  IndexType index;
  
  index[0] = pixelX;
  index[1] = pixelY;
  
  typedef ImageType::PixelType PixelType;
  PixelType pixel;
  
  pixel = relaxFilter->GetOutput()->GetPixel(index);		
  
  if (fabs(pixel - expectedValue) > tolerance) 
    {
    	std::cerr << "Expected value:" << expectedValue << ", but got:" << pixel << std::endl;
    	return EXIT_FAILURE;
    }
  
  // We are done. Go for coffee.
  return EXIT_SUCCESS;    
}
