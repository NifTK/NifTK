/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUltrasoundPointerBasedCalibration.h"
#include "niftkUltrasoundPointerCalibrationCostFunction.h"
#include <niftkArunLeastSquaresPointRegistration.h>
#include <mitkExceptionMacro.h>
#include <mitkPointUtils.h>
#include <mitkOpenCVMaths.h>
#include <itkLevenbergMarquardtOptimizer.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

//-----------------------------------------------------------------------------
UltrasoundPointerBasedCalibration::UltrasoundPointerBasedCalibration()
{
  m_ScalingMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_ScalingMatrix->Identity();
  m_RigidBodyMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_RigidBodyMatrix->Identity();
  m_UltrasoundImagePoints = mitk::PointSet::New();
  m_SensorPoints = mitk::PointSet::New();
  this->Modified();
}


//-----------------------------------------------------------------------------
UltrasoundPointerBasedCalibration::~UltrasoundPointerBasedCalibration()
{
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> UltrasoundPointerBasedCalibration::GetCalibrationMatrix() const
{
  vtkSmartPointer<vtkMatrix4x4> rigid = vtkSmartPointer<vtkMatrix4x4>::New();
  rigid->DeepCopy(m_RigidBodyMatrix);

  vtkSmartPointer<vtkMatrix4x4> scaling = vtkSmartPointer<vtkMatrix4x4>::New();
  scaling->DeepCopy(m_ScalingMatrix);

  vtkSmartPointer<vtkMatrix4x4> calibration = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkMatrix4x4::Multiply4x4(rigid, scaling, calibration);

  return calibration;
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> UltrasoundPointerBasedCalibration::GetRigidBodyMatrix() const
{
  vtkSmartPointer<vtkMatrix4x4> clone = vtkSmartPointer<vtkMatrix4x4>::New();
  clone->DeepCopy(m_RigidBodyMatrix);
  return clone;
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> UltrasoundPointerBasedCalibration::GetScalingMatrix() const
{
  vtkSmartPointer<vtkMatrix4x4> clone = vtkSmartPointer<vtkMatrix4x4>::New();
  clone->DeepCopy(m_ScalingMatrix);
  return clone;
}


//-----------------------------------------------------------------------------
void UltrasoundPointerBasedCalibration::SetSensorPoints(mitk::PointSet::Pointer points)
{
  m_SensorPoints = points;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPointerBasedCalibration::SetImagePoints(mitk::PointSet::Pointer points)
{
  m_UltrasoundImagePoints = points;
  this->Modified();
}


//-----------------------------------------------------------------------------
double UltrasoundPointerBasedCalibration::DoPointerBasedCalibration()
{
  MITK_INFO << "Doing DoPointerBasedCalibration(), with "
            << m_UltrasoundImagePoints->GetSize() << ", image points and "
            << m_SensorPoints->GetSize() << " sensor points";

  if (m_SensorPoints->GetSize() < 3)
  {
    mitkThrow() << "We have < 3 sensor points";
  }
  if (m_UltrasoundImagePoints->GetSize() < 3)
  {
    mitkThrow() << "We have < 3 image points";
  }
  if (m_UltrasoundImagePoints->GetSize() != m_SensorPoints->GetSize())
  {
    mitkThrow() << "We have a different number of ultrasound and sensor points";
  }

  // Take a guess at the relative scale.
  double scaleOfImagePoints = mitk::FindLargestDistanceBetweenTwoPoints(*m_UltrasoundImagePoints);
  if (fabs(scaleOfImagePoints) < 0.001)
  {
    mitkThrow() << "Image points too close together";
  }
  double scaleOfSensorPoints = mitk::FindLargestDistanceBetweenTwoPoints(*m_SensorPoints);
  if (fabs(scaleOfSensorPoints) < 0.001)
  {
    mitkThrow() << "Sensor points too close together";
  }

  double millimetresPerPixel = scaleOfSensorPoints/scaleOfImagePoints;

  // Now scale the image points.
  mitk::PointSet::Pointer scaledImagePoints = mitk::PointSet::New();
  mitk::ScalePointSets(*m_UltrasoundImagePoints, *scaledImagePoints, millimetresPerPixel);

  // Run SVD based registration, which throws mitk::Exception on error.
  vtkSmartPointer<vtkMatrix4x4> regMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  double fiducialRegistrationError = niftk::PointBasedRegistrationUsingSVD(
                                     m_SensorPoints, scaledImagePoints, *regMatrix);
  std::cout << "UltrasoundPointerBasedCalibration: scaling=" << millimetresPerPixel
            << ", SVD FRE=" << fiducialRegistrationError << std::endl;

  // Extract starting parameters for optimisation.
  mitk::Point3D rodriguesRotationParameters;
  mitk::Point3D translationParameters;
  mitk::ExtractRigidBodyParameters(*regMatrix, rodriguesRotationParameters, translationParameters);

  // Now optimise the scaling and rigid parameters.

  niftk::UltrasoundPointerCalibrationCostFunction::Pointer costFunction
      = niftk::UltrasoundPointerCalibrationCostFunction::New();

  niftk::UltrasoundPointerCalibrationCostFunction::ParametersType parameters;
  parameters.SetSize(costFunction->GetNumberOfParameters());
  parameters[0] = rodriguesRotationParameters[0];
  parameters[1] = rodriguesRotationParameters[1];
  parameters[2] = rodriguesRotationParameters[2];
  parameters[3] = translationParameters[0];
  parameters[4] = translationParameters[1];
  parameters[5] = translationParameters[2];
  parameters[6] = millimetresPerPixel;
  parameters[7] = millimetresPerPixel;

  std::cerr << "UltrasoundPointerBasedCalibration: Optimisation started at:" << parameters << std::endl;

  niftk::UltrasoundPointerCalibrationCostFunction::ParametersType scaleFactors;
  scaleFactors.SetSize(costFunction->GetNumberOfParameters());
  scaleFactors.Fill(1);

  costFunction->SetImagePoints(m_UltrasoundImagePoints);
  costFunction->SetSensorPoints(m_SensorPoints);
  costFunction->SetScales(scaleFactors);

  double residualError = 0;
  itk::LevenbergMarquardtOptimizer::Pointer optimizer = itk::LevenbergMarquardtOptimizer::New();
  optimizer->UseCostFunctionGradientOff(); // use default VNL derivative, not our one.
  optimizer->SetCostFunction(costFunction);
  optimizer->SetInitialPosition(parameters);
  optimizer->SetNumberOfIterations(20000000);
  optimizer->SetGradientTolerance(0.000000005);
  optimizer->SetEpsilonFunction(0.000000005);
  optimizer->SetValueTolerance(0.000000005);

  optimizer->StartOptimization();
  parameters = optimizer->GetCurrentPosition();

  niftk::UltrasoundPointerCalibrationCostFunction::MeasureType values = costFunction->GetValue(parameters);
  residualError = costFunction->GetResidual(values);

  std::cerr << "UltrasoundPointerBasedCalibration: Optimisation finished at:" << parameters
            << ", residual=" << residualError << std::endl;

  // Setup the output.
  m_ScalingMatrix = costFunction->GetScalingMatrix(parameters);
  m_RigidBodyMatrix = costFunction->GetRigidMatrix(parameters);

  // Finished.
  this->Modified();
  return residualError;
}

//-----------------------------------------------------------------------------
} // end namespace
