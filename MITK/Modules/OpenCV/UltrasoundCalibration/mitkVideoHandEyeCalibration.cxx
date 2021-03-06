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
  double residual = 0;

  bool doLag = this->GetOptimiseTimingLag();
  bool doRigid = this->GetOptimiseRigidTransformation();
  bool doInvariant = this->GetOptimiseInvariantPoint();

  if (doLag)
  {
    this->SetOptimiseTimingLag(false);
  }

  residual = this->DoCalibration();

  if (doLag)
  {
    this->SetOptimiseTimingLag(true);
    this->SetOptimiseRigidTransformation(false);
    this->SetOptimiseInvariantPoint(false);
    residual = this->DoCalibration();
    this->SetOptimiseTimingLag(doLag);
    this->SetOptimiseRigidTransformation(doRigid);
    this->SetOptimiseInvariantPoint(doInvariant);
  }

  return residual;
}


//-----------------------------------------------------------------------------
double VideoHandEyeCalibration::DoCalibration()
{
  if ( m_PointData == NULL )
  {
    mitkThrow() << "mitkVideoHandeyeCalibration::DoCalibration(): No point data available";
  }
  if ( m_TrackingData == NULL )
  {
    mitkThrow() << "mitkVideoHandeyeCalibration::DoCalibration(): No tracking data available";
  }

  double residualError = 0;

  itk::VideoHandEyeCalibrationCostFunction::ParametersType parameters;
  itk::VideoHandEyeCalibrationCostFunction::ParametersType scaleFactorsForCostFunctionDerivative;
  itk::VideoHandEyeCalibrationCostFunction::ParametersType scaleFactorsForParameterSizes;

  // Setup size of parameters array.
  int numberOfParameters = 0;
  if (this->GetOptimiseRigidTransformation())
  {
    numberOfParameters += 6;
  }
  if (this->GetOptimiseInvariantPoint())
  {
    numberOfParameters += 3;
  }
  if (this->GetOptimiseTimingLag())
  {
    numberOfParameters += 1;
  }
  assert(   numberOfParameters == 1
         || numberOfParameters == 6
         || numberOfParameters == 9
         || numberOfParameters == 10
         );

  parameters.SetSize(numberOfParameters);
  scaleFactorsForCostFunctionDerivative.SetSize(numberOfParameters);
  scaleFactorsForParameterSizes.SetSize(numberOfParameters);

  parameters.Fill(0);
  scaleFactorsForCostFunctionDerivative.Fill(0.1);

  if (this->GetOptimiseRigidTransformation())
  {
    std::vector<double> rigidParams = m_DownCastCostFunction->GetRigidTransformationParameters();
    parameters[0] = rigidParams[0];
    parameters[1] = rigidParams[1];
    parameters[2] = rigidParams[2];
    parameters[3] = rigidParams[3];
    parameters[4] = rigidParams[4];
    parameters[5] = rigidParams[5];

    scaleFactorsForParameterSizes[0] = 0.01;
    scaleFactorsForParameterSizes[1] = 0.01;
    scaleFactorsForParameterSizes[2] = 0.01;
    scaleFactorsForParameterSizes[3] = 1;
    scaleFactorsForParameterSizes[4] = 1;
    scaleFactorsForParameterSizes[5] = 1;
  }
  if (this->GetOptimiseInvariantPoint())
  {
    mitk::Point3D invariantPoint = this->GetInvariantPoint();
    parameters[6] = invariantPoint[0];
    parameters[7] = invariantPoint[1];
    parameters[8] = invariantPoint[2];

    scaleFactorsForParameterSizes[6] = 1;
    scaleFactorsForParameterSizes[7] = 1;
    scaleFactorsForParameterSizes[8] = 1;
  }
  if (this->GetOptimiseTimingLag())
  {
    double timeStamp = this->GetTimingLag();
    parameters[parameters.GetSize() -1] = timeStamp;

    scaleFactorsForParameterSizes[parameters.GetSize() -1] = 0.1;
  }

  MITK_INFO << "VideoHandEyeCalibration:Start parameters = " << parameters << std::endl;
  MITK_INFO << "VideoHandEyeCalibration:Optimising " << m_PointData->size() << " points and " << m_TrackingData->GetSize() << " matrices " << std::endl;

  m_DownCastCostFunction->SetPointData(m_PointData);
  m_DownCastCostFunction->SetTrackingData(m_TrackingData);
  m_DownCastCostFunction->SetNumberOfParameters(parameters.GetSize());
  m_DownCastCostFunction->SetScales(scaleFactorsForCostFunctionDerivative);

  itk::LevenbergMarquardtOptimizer::Pointer optimizer = itk::LevenbergMarquardtOptimizer::New();
  optimizer->UseCostFunctionGradientOff(); // use default VNL derivative, not our one.
  optimizer->SetCostFunction(m_DownCastCostFunction);
  optimizer->SetInitialPosition(parameters);
  optimizer->SetNumberOfIterations(20000000);
  optimizer->SetGradientTolerance(0.000000005);
  optimizer->SetEpsilonFunction(0.000000005);
  optimizer->SetValueTolerance(0.000000005);
  optimizer->SetScales(scaleFactorsForParameterSizes);

  optimizer->StartOptimization();
  parameters = optimizer->GetCurrentPosition();

  itk::VideoHandEyeCalibrationCostFunction::MeasureType values = m_DownCastCostFunction->GetValue(parameters);
  residualError = m_DownCastCostFunction->GetResidual(values);

  MITK_INFO << "Stop condition:" << optimizer->GetStopConditionDescription();

  if (this->GetOptimiseRigidTransformation())
  {
    std::vector<double> rigidParams;
    rigidParams.push_back(parameters[0]);
    rigidParams.push_back(parameters[1]);
    rigidParams.push_back(parameters[2]);
    rigidParams.push_back(parameters[3]);
    rigidParams.push_back(parameters[4]);
    rigidParams.push_back(parameters[5]);
    m_DownCastCostFunction->SetRigidTransformationParameters(rigidParams);
  }
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
    double timeStamp = parameters[parameters.GetSize() -1];
    this->SetTimingLag(timeStamp);
  }

  std::cout << "Calibration Summary:"
            << m_PointData->size() << ", "
            << residualError << ", "
            << parameters[0] << ", "
            << parameters[1] << ", "
            << parameters[2] << ", "
            << parameters[3] << ", "
            << parameters[4] << ", "
            << parameters[5] << ", ";
  if (this->GetOptimiseInvariantPoint())
  {
    std::cout << parameters[6] << ", "
              << parameters[7] << ", "
              << parameters[8] << ", " ;
  }
  std::cout << std::endl;

  return residualError;
}

//-----------------------------------------------------------------------------
} // end namespace
