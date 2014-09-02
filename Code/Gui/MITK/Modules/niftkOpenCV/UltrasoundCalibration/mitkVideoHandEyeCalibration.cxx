/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkVideoHandEyeCalibration.h"
#include <itkVideoHandEyeCalibrationCostFunction.h>
#include <itkLevenbergMarquardtOptimizer.h>
#include <cassert>

namespace mitk {

//-----------------------------------------------------------------------------
VideoHandEyeCalibration::VideoHandEyeCalibration()
{
  m_CostFunction = itk::VideoHandEyeCalibrationCostFunction::New();
  m_DownCastCostFunction = dynamic_cast<itk::VideoHandEyeCalibrationCostFunction*>(m_CostFunction.GetPointer());
  assert(m_DownCastCostFunction);
  this->Modified();
}


//-----------------------------------------------------------------------------
VideoHandEyeCalibration::~VideoHandEyeCalibration()
{
}


//-----------------------------------------------------------------------------
double VideoHandEyeCalibration::Calibrate()
{
  double residualError = 0;

  itk::VideoHandEyeCalibrationCostFunction::ParametersType parameters;
  itk::VideoHandEyeCalibrationCostFunction::ParametersType scaleFactors;

  // Setup size of parameters array.
  int numberOfParameters = 6;
  if (this->GetOptimiseInvariantPoint())
  {
    numberOfParameters += 3;
  }
  if (this->GetOptimiseTimingLag())
  {
    numberOfParameters += 1;
  }
  assert(numberOfParameters == 6
         || numberOfParameters == 9
         || numberOfParameters == 10
         );

  parameters.SetSize(numberOfParameters);
  scaleFactors.SetSize(numberOfParameters);

  parameters.Fill(0);
  scaleFactors.Fill(0.0000001);

  parameters[0] = m_RigidTransformation[0];
  parameters[1] = m_RigidTransformation[1];
  parameters[2] = m_RigidTransformation[2];
  parameters[3] = m_RigidTransformation[3];
  parameters[4] = m_RigidTransformation[4];
  parameters[5] = m_RigidTransformation[5];

  if (this->GetOptimiseInvariantPoint())
  {
    mitk::Point3D invariantPoint = this->GetInvariantPoint();
    parameters[6] = invariantPoint[0];
    parameters[7] = invariantPoint[1];
    parameters[8] = invariantPoint[2];
  }
  if (this->GetOptimiseTimingLag())
  {
    TimeStampType timeStamp = this->GetTimingLag();
    parameters[9] = timeStamp;
  }

  std::cout << "VideoHandEyeCalibration:Start parameters = " << parameters << std::endl;

  m_CostFunction->SetNumberOfParameters(parameters.GetSize());
  m_CostFunction->SetScales(scaleFactors);

  itk::LevenbergMarquardtOptimizer::Pointer optimizer = itk::LevenbergMarquardtOptimizer::New();
  optimizer->UseCostFunctionGradientOn();
  optimizer->SetCostFunction(m_CostFunction);
  optimizer->SetInitialPosition(parameters);
  optimizer->SetScales(scaleFactors);
  optimizer->SetNumberOfIterations(20000000);
  optimizer->SetGradientTolerance(0.0000005);
  optimizer->SetEpsilonFunction(0.0000005);
  optimizer->SetValueTolerance(0.0000005);

  optimizer->StartOptimization();

  parameters = optimizer->GetCurrentPosition();

  m_RigidTransformation[0] = parameters[0];
  m_RigidTransformation[1] = parameters[1];
  m_RigidTransformation[2] = parameters[2];
  m_RigidTransformation[3] = parameters[3];
  m_RigidTransformation[4] = parameters[4];
  m_RigidTransformation[5] = parameters[5];

  if (this->GetOptimiseInvariantPoint())
  {
    mitk::Point3D invariantPoint;
    invariantPoint[0] = parameters[6];
    invariantPoint[1] = parameters[7];
    invariantPoint[2] = parameters[8];
    this->SetInvariantPoint(invariantPoint);
  }
  if (this->GetOptimiseTimingLag())
  {
    TimeStampType timeStamp;
    timeStamp = parameters[9];
    this->SetTimingLag(timeStamp);
  }

  itk::VideoHandEyeCalibrationCostFunction::MeasureType values = m_CostFunction->GetValue(parameters);
  residualError = m_CostFunction->GetResidual(values);

  std::cout << "Stop condition:" << optimizer->GetStopConditionDescription();
  std::cout << "VideoHandEyeCalibration:End parameters = " << parameters << std::endl;
  std::cout << "VideoHandEyeCalibration:End residual = " << residualError << std::endl;

  return residualError;
}

//-----------------------------------------------------------------------------
} // end namespace
