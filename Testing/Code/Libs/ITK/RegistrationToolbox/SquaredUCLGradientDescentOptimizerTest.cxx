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
#include <ConversionUtils.h>
#include <itkEulerAffineTransform.h>
#include <itkArray.h>
#include <itkArray2D.h>
#include <itkPoint.h>
#include <itkGradientDescentOptimizer.h>
#include <itkIterationUpdateCommand.h>
#include <itkTranslationTransform.h>
#include "itkSquaredFunctionImageToImageMetric.h"
#include <itkArray.h>

int SquaredUCLGradientDescentOptimizerTest(int argc, char * argv[])
{

  if( argc < 8)
    {
    std::cerr << "Usage   : SquaredUCLGradientDescentOptimizerTest startX startY learningRate maxIterations resultTolerance finishX finishY" << std::endl;
    return 1;
    }
  double startX = niftk::ConvertToDouble(argv[1]);
  double startY = niftk::ConvertToDouble(argv[2]);
  double learningRate = niftk::ConvertToDouble(argv[3]);
  int maxIterations = niftk::ConvertToInt(argv[4]);
  double resultTolerance = niftk::ConvertToDouble(argv[5]);
  double finishX = niftk::ConvertToDouble(argv[6]);
  double finishY = niftk::ConvertToDouble(argv[7]);
  
  const unsigned int Dimension = 2;
  typedef itk::Image< short, Dimension>                                ImageType;
  typedef itk::TranslationTransform<double, 2>                         TransformType;
  typedef itk::GradientDescentOptimizer                                OptimizerType;
  typedef itk::SquaredFunctionImageToImageMetric<ImageType, ImageType> CostFunctionType;
  typedef itk::IterationUpdateCommand                                  TrackerType;
  typedef itk::Array<double>                                           ParametersType;
  
  OptimizerType::Pointer optimizer = OptimizerType::New();
  CostFunctionType::Pointer cost = CostFunctionType::New();
  TransformType::Pointer transform = TransformType::New();
  TrackerType::Pointer tracker = TrackerType::New();
 
  ParametersType parameters(2);
  parameters[0] = startX;
  parameters[1] = startY;
  
  cost->SetTransform(transform);
  
  optimizer->SetCostFunction(cost);
  optimizer->SetNumberOfIterations(maxIterations);
  optimizer->SetLearningRate(learningRate);
  optimizer->SetInitialPosition(parameters);
  optimizer->AddObserver( itk::IterationEvent(), tracker );
  optimizer->SetMinimize(true);    
  optimizer->StartOptimization();
  
  parameters = optimizer->GetCurrentPosition();
  
  std::cerr << "Final position:" << parameters << std::endl;
  
  if (fabs(parameters[0] - finishX) > resultTolerance) 
    {
      return EXIT_FAILURE;
    }

  if (fabs(parameters[1] - finishY) > resultTolerance) 
    {
      return EXIT_FAILURE;
    }
  
  return EXIT_SUCCESS;
}
