/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkUltrasoundPinCalibration.h"
#include <itkLevenbergMarquardtOptimizer.h>
#include <mitkExceptionMacro.h>
#include <set>

namespace mitk {

//-----------------------------------------------------------------------------
UltrasoundPinCalibration::~UltrasoundPinCalibration()
{
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibration::UltrasoundPinCalibration()
: m_OptimiseInvariantPoints(false)
{
  m_CostFunction = itk::UltrasoundPinCalibrationCostFunction::New();
  this->SetNumberOfInvariantPoints(1);
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibration::SetNumberOfInvariantPoints(const unsigned int& numberOfPoints)
{
  m_CostFunction->SetNumberOfInvariantPoints(numberOfPoints);
  if (numberOfPoints > 1)
  {
    this->SetRetrievePointIdentifier(true);
  }
  else
  {
    this->SetRetrievePointIdentifier(false);
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibration::InitialiseInvariantPoint(const std::vector<float>& commandLineArgs)
{
  if (commandLineArgs.size() == 3)
  {
    this->InitialiseInvariantPoint(0, commandLineArgs);
  }
  else if (commandLineArgs.size() == 4)
  {
    std::vector<float> tmp;
    tmp.push_back(commandLineArgs[1]);
    tmp.push_back(commandLineArgs[2]);
    tmp.push_back(commandLineArgs[3]);
    this->InitialiseInvariantPoint(static_cast<int>(commandLineArgs[0]), tmp);
  }
  else
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibration::InitialiseInvariantPoint given a commandLineArgs with the wrong number of elements, it should be 3 or 4." << std::endl;
    mitkThrow() << oss.str();
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibration::InitialiseInvariantPoint(const int &pointNumber, const std::vector<float>& commandLineArgs)
{
  if (commandLineArgs.size() != 3)
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibration::InitialiseInvariantPoint given a commandLineArgs with the wrong number of elements, it should be 3." << std::endl;
    mitkThrow() << oss.str();
  }

  mitk::Point3D point;
  point[0] = commandLineArgs[0];
  point[1] = commandLineArgs[1];
  point[2] = commandLineArgs[2];
  this->m_CostFunction->SetInvariantPoint(pointNumber, point);
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
  
  // Check number of invariant points.
  int numberOfInvariantPoints = 0;
  std::set<int> invariantPointIdentifiers;
  for (unsigned int i = 0; i < points.size(); i++)
  {
    invariantPointIdentifiers.insert(points[i].first);
  }
  numberOfInvariantPoints = invariantPointIdentifiers.size();
  if (numberOfInvariantPoints != m_CostFunction->GetNumberOfInvariantPoints())
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibration::Calibrate calculated numberOfInvariantPoints=" << numberOfInvariantPoints << ", but cost function says it should be=" << m_CostFunction->GetNumberOfInvariantPoints() << std::endl;
    mitkThrow() << oss.str();
  }

  // Setup size of parameters array.
  int numberOfParameters = 6;
  if (this->m_OptimiseScaling)
  {
    numberOfParameters += 2;
  }
  if (m_OptimiseInvariantPoints)
  {
    numberOfParameters += (numberOfInvariantPoints*3);
  }
  parameters.SetSize(numberOfParameters);
  scaleFactors.SetSize(numberOfParameters);

  if (this->m_OptimiseScaling)
  {
    parameters[6] = this->m_MillimetresPerPixel[0];
    parameters[7] = this->m_MillimetresPerPixel[1];
    scaleFactors[6] = 0.0001;
    scaleFactors[7] = 0.0001;
  }

  if (m_OptimiseInvariantPoints)
  {
    int offset = 6;
    if (this->m_OptimiseScaling)
    {
      offset = 8;
    }
    // For now, initialise all invariant points to the same initial guess.
    for (int i = 0; i < numberOfInvariantPoints; i++)
    {
      parameters[offset + 3*i + 0] = m_CostFunction->GetInvariantPoint(i)[0];
      parameters[offset + 3*i + 1] = m_CostFunction->GetInvariantPoint(i)[1];
      parameters[offset + 3*i + 2] = m_CostFunction->GetInvariantPoint(i)[2];
      scaleFactors[offset + 3*i + 0] = 0.001;
      scaleFactors[offset + 3*i + 1] = 0.001;
      scaleFactors[offset + 3*i + 2] = 0.001;
    }
  }

  parameters[0] = this->m_InitialGuess[0];
  parameters[1] = this->m_InitialGuess[1];
  parameters[2] = this->m_InitialGuess[2];
  parameters[3] = this->m_InitialGuess[3];
  parameters[4] = this->m_InitialGuess[4];
  parameters[5] = this->m_InitialGuess[5];

  scaleFactors[0] = 0.01;
  scaleFactors[1] = 0.01;
  scaleFactors[2] = 0.01;
  scaleFactors[3] = 0.001;
  scaleFactors[4] = 0.001;
  scaleFactors[5] = 0.001;
  
  std::cout << "UltrasoundPinCalibration:Start parameters = " << parameters << std::endl;
  std::cout << "UltrasoundPinCalibration:Start scale factors = " << scaleFactors << std::endl;
  
  m_CostFunction->SetMatrices(matrices);
  m_CostFunction->SetPoints(points);
  m_CostFunction->SetScales(scaleFactors);
  m_CostFunction->SetNumberOfParameters(parameters.GetSize());
  m_CostFunction->SetMillimetresPerPixel(this->m_MillimetresPerPixel);

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
  outputMatrix = m_CostFunction->GetCalibrationTransformation(parameters);

  std::cout << "Stop condition:" << optimizer->GetStopConditionDescription();
  std::cout << "UltrasoundPinCalibration:End parameters = " << parameters << std::endl;

  if (this->m_OptimiseScaling)
  {
    this->m_MillimetresPerPixel[0] = parameters[6];
    this->m_MillimetresPerPixel[1] = parameters[7];
  }

  itk::UltrasoundPinCalibrationCostFunction::MeasureType values = m_CostFunction->GetValue(parameters);
  residualError = m_CostFunction->GetResidual(values);

  return residualError;
}

//-----------------------------------------------------------------------------
} // end namespace
