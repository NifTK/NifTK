/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkUltrasoundPointerCalibration.h"
#include <itkUltrasoundPointerCalibrationCostFunction.h>
#include <itkLevenbergMarquardtOptimizer.h>
#include <mitkExceptionMacro.h>
#include <niftkVTKFunctions.h>

namespace mitk {

//-----------------------------------------------------------------------------
UltrasoundPointerCalibration::UltrasoundPointerCalibration()
: m_PointerTrackerToProbeTrackerTransform(NULL)
, m_ProbeToProbeTrackerTransform(NULL)
{
  m_PointerOffset[0] = 0;
  m_PointerOffset[1] = 0;
  m_PointerOffset[2] = 0;
  m_PointerTrackerToProbeTrackerTransform = vtkMatrix4x4::New();
  m_PointerTrackerToProbeTrackerTransform->Identity();
  m_ProbeToProbeTrackerTransform = vtkMatrix4x4::New();
  m_ProbeToProbeTrackerTransform->Identity();
}


//-----------------------------------------------------------------------------
UltrasoundPointerCalibration::~UltrasoundPointerCalibration()
{
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibration::InitialisePointerOffset(const std::vector<float>& commandLineArgs)
{
  if (commandLineArgs.size() == 3)
  {
    m_PointerOffset[0] = commandLineArgs[0];
    m_PointerOffset[1] = commandLineArgs[1];
    m_PointerOffset[2] = commandLineArgs[2];
  }
  else
  {
    m_PointerOffset[0] = 0;
    m_PointerOffset[1] = 0;
    m_PointerOffset[2] = 0;
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibration::InitialisePointerTrackerToProbeTrackerTransform(const std::string& fileName)
{
  if(fileName.size() != 0)
  {
    m_PointerTrackerToProbeTrackerTransform = niftk::LoadMatrix4x4FromFile(fileName, false);
  }
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibration::InitialiseProbeToProbeTrackerTransform(const std::string& fileName)
{
  if(fileName.size() != 0)
  {
    m_ProbeToProbeTrackerTransform = niftk::LoadMatrix4x4FromFile(fileName, false);
  }
}


//-----------------------------------------------------------------------------
double UltrasoundPointerCalibration::Calibrate(const std::vector< cv::Mat >& matrices,
    const std::vector<std::pair<int, cv::Point2d> > &points,
    cv::Matx44d& outputMatrix
    )
{
  double residualError = 0;

  itk::UltrasoundPointerCalibrationCostFunction::ParametersType parameters;
  itk::UltrasoundPointerCalibrationCostFunction::ParametersType scaleFactors;
  
  if (!this->m_OptimiseScaling)
  {
    parameters.SetSize(6);
    scaleFactors.SetSize(6);
  }
  else if (this->m_OptimiseScaling)
  {
    parameters.SetSize(8);
    parameters[6] = this->m_MillimetresPerPixel[0];
    parameters[7] = this->m_MillimetresPerPixel[1];

    scaleFactors.SetSize(8);
    scaleFactors[6] = 0.0001;
    scaleFactors[7] = 0.0001;
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
  
  std::cout << "UltrasoundPointerCalibration:Start parameters = " << parameters << std::endl;
  std::cout << "UltrasoundPointerCalibration:Start scale factors = " << scaleFactors << std::endl;
  
  itk::UltrasoundPointerCalibrationCostFunction::Pointer costFunction = itk::UltrasoundPointerCalibrationCostFunction::New();
  costFunction->SetMatrices(matrices);
  costFunction->SetPoints(points);
  costFunction->SetScales(scaleFactors);
  costFunction->SetNumberOfParameters(parameters.GetSize());
  costFunction->SetPointerOffset(m_PointerOffset);
  costFunction->SetMillimetresPerPixel(this->m_MillimetresPerPixel);
  costFunction->SetPointerTrackerToProbeTrackerTransform(*m_PointerTrackerToProbeTrackerTransform);
  costFunction->SetProbeToProbeTrackerTransform(*m_ProbeToProbeTrackerTransform);

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
    m_MillimetresPerPixel[0] = parameters[6];
    m_MillimetresPerPixel[1] = parameters[7];
  }

  itk::UltrasoundPointerCalibrationCostFunction::MeasureType values = costFunction->GetValue(parameters);
  residualError = costFunction->GetResidual(values);

  return residualError;
}

//-----------------------------------------------------------------------------
} // end namespace
