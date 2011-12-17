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

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include <iostream>
#include "itkCurvatureAnisotropicDiffusionImageFilter.h"
#include "itkWatershedImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkUnaryFunctorImageFilter.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSobelEdgeDetectionImageFilter.h"

int main( int argc, char *argv[] )
{
  if (argc < 7 )
  {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " inputImage outputImage conductanceTerm diffusionIterations lowerThreshold outputScaleLevel " << std::endl;
    return 1;
  }
  
  const unsigned int Dimensions = 3; 
  typedef short PixelType; 
  typedef itk::Image<PixelType, Dimensions> ImageType; 
  typedef itk::Image<unsigned long, Dimensions> LabeledImageType;
  typedef itk::Image<float, Dimensions> FloatImageType;

  typedef itk::ImageFileReader<ImageType> FileReaderType;
  FileReaderType::Pointer reader = FileReaderType::New();
  reader->SetFileName(argv[1]);
  
  typedef itk::CurvatureAnisotropicDiffusionImageFilter<ImageType, FloatImageType> DiffusionFilterType;
  DiffusionFilterType::Pointer diffusion = DiffusionFilterType::New();
  diffusion->SetNumberOfIterations( atoi(argv[4]) );
  diffusion->SetConductanceParameter( atof(argv[3]) );
  diffusion->SetTimeStep(0.0625);
  
  typedef itk::GradientMagnitudeImageFilter<FloatImageType, FloatImageType> GradientMagnitudeFilterType; 
  GradientMagnitudeFilterType::Pointer gradient = GradientMagnitudeFilterType::New();
  
  typedef itk::SobelEdgeDetectionImageFilter<FloatImageType, FloatImageType> SobelEdgeDetectionImageFilterType; 
  SobelEdgeDetectionImageFilterType::Pointer sobel = SobelEdgeDetectionImageFilterType::New(); 
  
  typedef itk::WatershedImageFilter<FloatImageType> WatershedFilterType;
  WatershedFilterType::Pointer watershed = WatershedFilterType::New();
  watershed->SetThreshold( atof(argv[5]) );
  watershed->SetLevel( atof(argv[6]) );
  
  typedef itk::CastImageFilter<WatershedFilterType::OutputImageType, ImageType> CastImageFilterType; 
  CastImageFilterType::Pointer castImageFilter = CastImageFilterType::New(); 
  
  typedef itk::ImageFileWriter<ImageType> FileWriterType;
  FileWriterType::Pointer writer = FileWriterType::New();
  writer->SetFileName(argv[2]);
  
  typedef itk::ImageFileWriter<FloatImageType> FloatFileWriterType;
  FloatFileWriterType::Pointer floatWriter = FloatFileWriterType::New(); 

  diffusion->SetInput(reader->GetOutput());
  
  gradient->SetInput(diffusion->GetOutput());
  sobel->SetInput(diffusion->GetOutput()); 
  
  watershed->SetInput(sobel->GetOutput());
  
  castImageFilter->SetInput(watershed->GetOutput()); 
  writer->SetInput(castImageFilter->GetOutput());

  try 
  {
    floatWriter->SetInput(diffusion->GetOutput()); 
    floatWriter->SetFileName("smooth.hdr"); 
    floatWriter->Update(); 
    
    floatWriter->SetInput(gradient->GetOutput()); 
    floatWriter->SetFileName("gradient.hdr"); 
    floatWriter->Update(); 
    
    floatWriter->SetInput(sobel->GetOutput()); 
    floatWriter->SetFileName("sobel.hdr"); 
    floatWriter->Update(); 
    
    writer->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }
    
  return 0;
}

