/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkUltrasoundPinCalibration.h"
#include <itkUltrasoundPinCalibrationCostFunction.h>
#include <itkLevenbergMarquardtOptimizer.h>
#include <cassert>

namespace mitk {

//-----------------------------------------------------------------------------
UltrasoundPinCalibration::UltrasoundPinCalibration()
{
  m_CostFunction = itk::UltrasoundPinCalibrationCostFunction::New();
  m_DownCastCostFunction = dynamic_cast<itk::UltrasoundPinCalibrationCostFunction*>(m_CostFunction.GetPointer());
  assert(m_DownCastCostFunction);
  this->Modified();
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibration::~UltrasoundPinCalibration()
{
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibration::SetImageScaleFactors(const mitk::Point2D& point)
{
  m_DownCastCostFunction->SetScaleFactors(point);
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::Point2D UltrasoundPinCalibration::GetImageScaleFactors() const
{
  return m_DownCastCostFunction->GetScaleFactors();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibration::SetOptimiseImageScaleFactors(const bool& optimise)
{
  m_DownCastCostFunction->SetOptimiseScaleFactors(optimise);
  this->Modified();
}


//-----------------------------------------------------------------------------
bool UltrasoundPinCalibration::GetOptimiseImageScaleFactors() const
{
  return m_DownCastCostFunction->GetOptimiseScaleFactors();
}


//-----------------------------------------------------------------------------
double UltrasoundPinCalibration::Calibrate()
{
  assert(m_PointData);
  assert(m_TrackingData);

  double residualError = 0;

  itk::UltrasoundPinCalibrationCostFunction::ParametersType parameters;
  itk::UltrasoundPinCalibrationCostFunction::ParametersType scaleFactors;
  
  // Setup size of parameters array.
  int numberOfParameters = 0;
  if (this->GetOptimiseRigidTransformation())
  {
    numberOfParameters += 6;
  }
  if (this->GetOptimiseImageScaleFactors())
  {
    numberOfParameters += 2;
  }
  if (this->GetOptimiseInvariantPoint())
  {
    numberOfParameters += 3;
  }
  if (this->GetOptimiseTimingLag())
  {
    numberOfParameters += 1;
  }
  
  if ( this->GetOptimiseTimingLag() && ( numberOfParameters != 1 || numberOfParameters != 12 ) )
  {
    MITK_ERROR << "You can't optimise timing lag without all or none of the other parameters";
  }

  assert(   numberOfParameters == 1
         || numberOfParameters == 6
         || numberOfParameters == 9
         || numberOfParameters == 11
         || numberOfParameters == 12
         );

  parameters.SetSize(numberOfParameters);
  scaleFactors.SetSize(numberOfParameters);

  parameters.Fill(0);
  scaleFactors.Fill(0.1);

  if (this->GetOptimiseRigidTransformation())
  {
    std::vector<double> rigidParams = m_DownCastCostFunction->GetRigidTransformationParameters();
    parameters[0] = rigidParams[0];
    parameters[1] = rigidParams[1];
    parameters[2] = rigidParams[2];
    parameters[3] = rigidParams[3];
    parameters[4] = rigidParams[4];
    parameters[5] = rigidParams[5];
  }
  if (this->GetOptimiseInvariantPoint())
  {
    mitk::Point3D invariantPoint = this->GetInvariantPoint();
    parameters[6] = invariantPoint[0];
    parameters[7] = invariantPoint[1];
    parameters[8] = invariantPoint[2];
  }
  if (this->GetOptimiseImageScaleFactors())
  {
    mitk::Point2D scaleFactors = this->GetImageScaleFactors();
    parameters[9] = scaleFactors[0];
    parameters[10] = scaleFactors[1];
  }
  if (this->GetOptimiseTimingLag())
  {
    double timeStamp = this->GetTimingLag();
    parameters[11] = timeStamp;
  }
  
  std::cout << "UltrasoundPinCalibration:Start parameters = " << parameters << std::endl;
  std::cout << "UltrasoundPinCalibration:Optimising " << m_PointData->size() << " points and " << m_TrackingData->GetSize() << " matrices " << std::endl;

  m_DownCastCostFunction->SetPointData(m_PointData);
  m_DownCastCostFunction->SetTrackingData(m_TrackingData);
  m_DownCastCostFunction->SetNumberOfParameters(parameters.GetSize());
  m_DownCastCostFunction->SetScales(scaleFactors);

  itk::LevenbergMarquardtOptimizer::Pointer optimizer = itk::LevenbergMarquardtOptimizer::New();
  optimizer->UseCostFunctionGradientOn(); // use default VNL derivative, not our one.
  optimizer->SetCostFunction(m_DownCastCostFunction);
  optimizer->SetInitialPosition(parameters);
  optimizer->SetNumberOfIterations(20000000);
  optimizer->SetGradientTolerance(0.000000005);
  optimizer->SetEpsilonFunction(0.000000005);
  optimizer->SetValueTolerance(0.000000005);

  optimizer->StartOptimization();
  parameters = optimizer->GetCurrentPosition();

  itk::UltrasoundPinCalibrationCostFunction::MeasureType values = m_DownCastCostFunction->GetValue(parameters);
  residualError = m_DownCastCostFunction->GetResidual(values);

  std::cout << "Stop condition:" << optimizer->GetStopConditionDescription();

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
  if (this->GetOptimiseImageScaleFactors())
  {
    mitk::Point2D scaleFactors;
    scaleFactors[0] = parameters[9];
    scaleFactors[1] = parameters[10];
    this->SetImageScaleFactors(scaleFactors);
  }
  if (this->GetOptimiseTimingLag())
  {
    double timeStamp;
    timeStamp = parameters[9];
    this->SetTimingLag(timeStamp);
  }

  return residualError;
}

//-----------------------------------------------------------------------------
} // end namespace
