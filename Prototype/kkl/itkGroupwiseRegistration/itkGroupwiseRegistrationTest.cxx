/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkNMIGroupwiseImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkPowellOptimizer.h"
#include "itkAffineTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkTranslationTransform.h"

const unsigned int Dimension = 2;
typedef double PixelType;
typedef itk::Image< PixelType, Dimension >  FixedImageType;

int main(int argc, char* argv[])
{
  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::LinearInterpolateImageFunction< FixedImageType, double> InterpolatorType;
  //typedef itk::AffineTransform< double,Dimension > TransformType;
  typedef itk::TranslationTransform< double,Dimension > TransformType;
  typedef itk::PowellOptimizer OptimizerType; 
	typedef itk::NMIGroupwiseImageToImageMetric< FixedImageType > MetricType;  
	MetricType::Pointer metric = MetricType::New();
	std::vector< FixedImageReaderType::Pointer > imageReaders;
	std::vector< InterpolatorType::Pointer > interpolators;  
	std::vector< TransformType::Pointer > transforms;
  std::vector< TransformType::ParametersType > parametersGroup;
	
	for (int i = 1; i < argc; i++)
	{
    // Read in the images.  
	 	FixedImageReaderType::Pointer tempImageReader = FixedImageReaderType::New();
		
		tempImageReader->SetFileName(argv[i]);
		tempImageReader->Update();
		imageReaders.push_back(tempImageReader);
		
    // Set up interpolator and transform.   
		InterpolatorType::Pointer tempInterpolator = InterpolatorType::New();
		TransformType::Pointer tempTransform = TransformType::New();
		
		interpolators.push_back(tempInterpolator);
		transforms.push_back(tempTransform);
		
		metric->AddImage(tempImageReader->GetOutput(), tempTransform.GetPointer(), tempInterpolator.GetPointer());
    
    // Store initial transform parameters. 
    tempTransform->SetIdentity();
    parametersGroup.push_back(tempTransform->GetParameters());
	}	
  
	// Initialise the metirc group.
	metric->InitialiseMetricGroup();
	
  // Set up the optimizer.  
  unsigned int maxNumberOfIterations = 3;
  unsigned int numberOfIterations = 5;
  double stepLength = 1.0;
  OptimizerType::Pointer optimizer = OptimizerType::New();
  OptimizerType::ScalesType scales(transforms[0]->GetNumberOfParameters());
  std::vector< FixedImageType::Pointer > currentImageGroup(imageReaders.size()); 
  
  optimizer->DebugOn();
  
  scales.Fill(1.0);
  optimizer->SetCostFunction(metric);
  optimizer->SetScales(scales);
  optimizer->SetMaximize(true);
  optimizer->SetMaximumIteration(numberOfIterations);
  optimizer->SetMaximumLineIteration(numberOfIterations);
  optimizer->SetStepLength(stepLength);
  optimizer->SetStepTolerance(0.01);
  optimizer->SetValueTolerance(0.0001);
	
  try
  { 
    for (unsigned int loopIndex = 0; loopIndex < maxNumberOfIterations; loopIndex++)
    {  
      for (unsigned int index = 0; index < parametersGroup.size(); index++)
      {
        // Optimise the transformation. 
        optimizer->SetInitialPosition(parametersGroup[index]);
        metric->SetCurrentMovingImageIndex(index);
        optimizer->StartOptimization();
        parametersGroup[index] = optimizer->GetCurrentPosition();
        
        // Update the image. 
        typedef itk::ResampleImageFilter< FixedImageType, FixedImageType > ResampleFilterType;
        ResampleFilterType::Pointer resampler = ResampleFilterType::New();
        typedef itk::BSplineInterpolateImageFunction< FixedImageType > BSplineInterpolatorType;
        BSplineInterpolatorType::Pointer bsplineInterpolator  = BSplineInterpolatorType::New();
        
        resampler->SetInput(imageReaders[index]->GetOutput());
        transforms[index]->SetParameters(parametersGroup[index]);
        resampler->SetTransform(transforms[index]);
        resampler->SetInterpolator(bsplineInterpolator);
        resampler->SetDefaultPixelValue(0.0);
        resampler->SetOutputOrigin(imageReaders[index]->GetOutput()->GetOrigin());
        resampler->SetOutputSpacing(imageReaders[index]->GetOutput()->GetSpacing());
        resampler->SetSize(imageReaders[index]->GetOutput()->GetLargestPossibleRegion().GetSize());
        resampler->Update();
        currentImageGroup[index] = resampler->GetOutput();
        currentImageGroup[index]->DisconnectPipeline();
        metric->UpdateFixedImage(currentImageGroup[index], index);
      }
    }
    
    for (unsigned int index = 0; index < parametersGroup.size(); index++)
    {
      typedef itk::ResampleImageFilter< FixedImageType, FixedImageType > ResampleFilterType;
      ResampleFilterType::Pointer resampler = ResampleFilterType::New();
      typedef itk::BSplineInterpolateImageFunction< FixedImageType > BSplineInterpolatorType;
      BSplineInterpolatorType::Pointer bsplineInterpolator  = BSplineInterpolatorType::New();
      typedef unsigned char OutputPixelType;
      typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
      typedef itk::ImageFileWriter< OutputImageType >  WriterType;
      typedef itk::RescaleIntensityImageFilter< FixedImageType, OutputImageType > RescalerType;
      RescalerType::Pointer intensityRescaler = RescalerType::New();
      WriterType::Pointer writer = WriterType::New();
      char outputFilename[100];
      
      sprintf(outputFilename, "output-%u.png", index);
      resampler->SetInput(imageReaders[index]->GetOutput());
      transforms[index]->SetParameters(parametersGroup[index]);
      resampler->SetTransform(transforms[index]);
      resampler->SetInterpolator(bsplineInterpolator);
      resampler->SetDefaultPixelValue(0.0);
      resampler->SetOutputOrigin(imageReaders[index]->GetOutput()->GetOrigin());
      resampler->SetOutputSpacing(imageReaders[index]->GetOutput()->GetSpacing());
      resampler->SetSize(imageReaders[index]->GetOutput()->GetLargestPossibleRegion().GetSize());
      resampler->Update();
      writer->SetFileName(outputFilename);
      intensityRescaler->SetInput(resampler->GetOutput());
      intensityRescaler->SetOutputMinimum(0);
      intensityRescaler->SetOutputMaximum(255);
      writer->SetInput(intensityRescaler->GetOutput());
      writer->Update();
    }
    
  }
  catch (itk::ExceptionObject & err) 
  { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
  }
 	
	
	return EXIT_SUCCESS;    
	
}
