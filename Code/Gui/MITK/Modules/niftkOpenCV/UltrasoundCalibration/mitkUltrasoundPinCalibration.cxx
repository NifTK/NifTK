/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkUltrasoundPinCalibration.h"
#include "itkUltrasoundPinCalibrationCostFunction.h"
#include <itkLevenbergMarquardtOptimizer.h>
#include <mitkExceptionMacro.h>

namespace mitk {

//-----------------------------------------------------------------------------
UltrasoundPinCalibration::UltrasoundPinCalibration()
: m_OptimiseInvariantPoint(false)
{
  m_InvariantPoint[0] = 0;
  m_InvariantPoint[1] = 0;
  m_InvariantPoint[2] = 0;
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibration::~UltrasoundPinCalibration()
{
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibration::InitialiseInvariantPoint(const std::vector<float>& commandLineArgs)
{
  if (commandLineArgs.size() == 3)
  {
    m_InvariantPoint[0] = commandLineArgs[0];
    m_InvariantPoint[1] = commandLineArgs[1];
    m_InvariantPoint[2] = commandLineArgs[2];
  }
  else
  {
    m_InvariantPoint[0] = 0;
    m_InvariantPoint[1] = 0;
    m_InvariantPoint[2] = 0;
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
double UltrasoundPinCalibration::Calibrate(const std::vector< cv::Mat >& matrices,
    const std::vector<std::pair<int, cv::Point2d> > &points,
    cv::Matx44d& outputMatrix
    )
{
  double residualError = 0;

  itk::UltrasoundPinCalibrationCostFunction::ParametersType parameters;
  itk::UltrasoundPinCalibrationCostFunction::ParametersType scaleFactors;
  
  if (!this->m_OptimiseScaling && !m_OptimiseInvariantPoint)
  {
    parameters.SetSize(6);
    scaleFactors.SetSize(6);
  }
  else if (!this->m_OptimiseScaling
           && m_OptimiseInvariantPoint
          )
  {
    parameters.SetSize(9);
    parameters[6] = m_InvariantPoint[0];
    parameters[7] = m_InvariantPoint[1];
    parameters[8] = m_InvariantPoint[2];
    
    scaleFactors.SetSize(9);
    scaleFactors[6] = 1;
    scaleFactors[7] = 1;
    scaleFactors[8] = 1;
  }
  else if (this->m_OptimiseScaling && !m_OptimiseInvariantPoint)
  {
    parameters.SetSize(8);
    parameters[6] = this->m_MillimetresPerPixel[0];
    parameters[7] = this->m_MillimetresPerPixel[1];

    scaleFactors.SetSize(8);
    scaleFactors[6] = 0.0001;
    scaleFactors[7] = 0.0001;
  }
  else if (this->m_OptimiseScaling
           && m_OptimiseInvariantPoint
          )
  {
    parameters.SetSize(11);
    parameters[6] = this->m_MillimetresPerPixel[0];
    parameters[7] = this->m_MillimetresPerPixel[1];
    parameters[7] = m_InvariantPoint[0];
    parameters[8] = m_InvariantPoint[1];
    parameters[9] = m_InvariantPoint[2];
    
    scaleFactors.SetSize(11);
    scaleFactors[6] = 0.0001;
    scaleFactors[7] = 0.0001;
    scaleFactors[8] = 1;
    scaleFactors[9] = 1;
    scaleFactors[10] = 1;
  }

  parameters[0] = this->m_InitialGuess[0];
  parameters[1] = this->m_InitialGuess[1];
  parameters[2] = this->m_InitialGuess[2];
  parameters[3] = this->m_InitialGuess[3];
  parameters[4] = this->m_InitialGuess[4];
  parameters[5] = this->m_InitialGuess[5];

  scaleFactors[0] = 0.1;
  scaleFactors[1] = 0.1;
  scaleFactors[2] = 0.1;
  scaleFactors[3] = 1;
  scaleFactors[4] = 1;
  scaleFactors[5] = 1;
  
  std::cout << "UltrasoundPinCalibration:Start parameters = " << parameters << std::endl;
  std::cout << "UltrasoundPinCalibration:Start scale factors = " << scaleFactors << std::endl;
  
  itk::UltrasoundPinCalibrationCostFunction::Pointer costFunction = itk::UltrasoundPinCalibrationCostFunction::New();
  costFunction->SetMatrices(matrices);
  costFunction->SetPoints(points);
  costFunction->SetScales(scaleFactors);
  costFunction->SetNumberOfParameters(parameters.GetSize());
  costFunction->SetInvariantPoint(m_InvariantPoint);
  costFunction->SetMillimetresPerPixel(this->m_MillimetresPerPixel);

  itk::LevenbergMarquardtOptimizer::Pointer optimizer = itk::LevenbergMarquardtOptimizer::New();
  optimizer->UseCostFunctionGradientOff();
  optimizer->SetCostFunction(costFunction);
  optimizer->SetInitialPosition(parameters);
  optimizer->SetScales(scaleFactors);
  optimizer->SetNumberOfIterations(20000000);
  optimizer->SetGradientTolerance(0.0000005);
  optimizer->SetEpsilonFunction(0.0000005);
  optimizer->SetValueTolerance(0.0000005);

  optimizer->StartOptimization();

  parameters = optimizer->GetCurrentPosition();
  outputMatrix = costFunction->GetCalibrationTransformation(parameters);

  std::cerr << "Stop condition:" << optimizer->GetStopConditionDescription();

  if (this->m_OptimiseScaling)
  {
    this->m_MillimetresPerPixel[0] = parameters[6];
    this->m_MillimetresPerPixel[1] = parameters[7];
  }
  if (!this->m_OptimiseScaling
      && m_OptimiseInvariantPoint
      )
  {
    m_InvariantPoint[0] = parameters[6];
    m_InvariantPoint[1] = parameters[7];
    m_InvariantPoint[2] = parameters[8];
  }
  if (this->m_OptimiseScaling && m_OptimiseInvariantPoint)
  {
    m_InvariantPoint[0] = parameters[8];
    m_InvariantPoint[1] = parameters[9];
    m_InvariantPoint[2] = parameters[10];
  }

  itk::UltrasoundPinCalibrationCostFunction::MeasureType values = costFunction->GetValue(parameters);
  residualError = costFunction->GetResidual(values);

  return residualError;
}

//-----------------------------------------------------------------------------
} // end namespace
