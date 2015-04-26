/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkUltrasoundPointerCalibrationCostFunction.h"
#include <mitkOpenCVMaths.h>
#include <mitkExceptionMacro.h>

namespace itk {

//-----------------------------------------------------------------------------
UltrasoundPointerCalibrationCostFunction::UltrasoundPointerCalibrationCostFunction()
: m_NumberOfParameters(8)
{
  m_Scales.SetSize(8);
  m_Scales.Fill(0);
  m_ImagePoints = mitk::PointSet::New();
  m_SensorPoints = mitk::PointSet::New();
}


//-----------------------------------------------------------------------------
UltrasoundPointerCalibrationCostFunction::~UltrasoundPointerCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
unsigned int UltrasoundPointerCalibrationCostFunction::GetNumberOfValues(void) const
{
  return m_ImagePoints->GetSize() * 3;
}


//-----------------------------------------------------------------------------
unsigned int UltrasoundPointerCalibrationCostFunction::GetNumberOfParameters(void) const
{
  return m_NumberOfParameters;
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibrationCostFunction::SetScales(const ParametersType& scales)
{
  m_Scales = scales;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibrationCostFunction::SetImagePoints(const mitk::PointSet::Pointer imagePoints)
{
  m_ImagePoints = imagePoints;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibrationCostFunction::SetSensorPoints(const mitk::PointSet::Pointer sensorPoints)
{
  m_SensorPoints = sensorPoints;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibrationCostFunction::ValidateSizeOfParametersArray(const ParametersType & parameters) const
{
  if (parameters.GetSize() != this->m_NumberOfParameters )
  {
    mitkThrow() << "UltrasoundPointerCalibrationCostFunction::ValidateSizeOfParametersArray given " << parameters.GetSize() << " parameters, but was expecting " << this->m_NumberOfParameters << " parameters." << std::endl;
  }
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibrationCostFunction::ValidateSizeOfScalesArray(const ParametersType & parameters) const
{
  if (parameters.GetSize() != this->m_NumberOfParameters )
  {
    mitkThrow() << "UltrasoundPointerCalibrationCostFunction::ValidateSizeOfScalesArray given " << parameters.GetSize() << " parameters, but was expecting " << this->m_NumberOfParameters << " parameters.";
  }
}


//-----------------------------------------------------------------------------
double UltrasoundPointerCalibrationCostFunction::GetResidual(const MeasureType& values) const
{
  double residual = 0;
  unsigned int numberOfValues = values.size();

  for (unsigned int i = 0; i < numberOfValues; i++)
  {
    residual += (values[i]*values[i]);
  }

  if (numberOfValues > 0)
  {
    residual /= static_cast<double>(numberOfValues);
  }

  residual = sqrt(residual);

  return residual;
}


//-----------------------------------------------------------------------------
void UltrasoundPointerCalibrationCostFunction::GetDerivative(
  const ParametersType& parameters, DerivativeType& derivative) const
{
  this->ValidateSizeOfScalesArray(parameters);

  MeasureType forwardValue;
  MeasureType backwardValue;

  ParametersType offsetParameters;
  derivative.SetSize(this->GetNumberOfParameters(), this->GetNumberOfValues());

  for (unsigned int i = 0; i < this->GetNumberOfParameters(); i++)
  {
    offsetParameters = parameters;
    offsetParameters[i] += m_Scales[i];
    forwardValue = this->GetValue(offsetParameters);

    offsetParameters = parameters;
    offsetParameters[i] -= m_Scales[i];
    backwardValue = this->GetValue(offsetParameters);

    for (unsigned int j = 0; j < this->GetNumberOfValues(); j++)
    {
      derivative[i][j] = (forwardValue[j] - backwardValue[j])/2.0;
    }
  }

  // Normalise (?)
  double norm = 0;
  for (unsigned int j = 0; j < this->GetNumberOfValues(); j++)
  {
    norm = 0;
    for (unsigned int i = 0; i < this->GetNumberOfParameters(); i++)
    {
      norm += (derivative[i][j]*derivative[i][j]);
    }
    if (norm != 0)
    {
      norm = sqrt(norm);

      for (unsigned int i = 0; i < this->GetNumberOfParameters(); i++)
      {
        derivative[i][j] = derivative[i][j]*m_Scales[i]/norm;
      }
    }
  }
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> UltrasoundPointerCalibrationCostFunction::GetRigidMatrix(const ParametersType & parameters) const
{
  cv::Matx44d scalingMatrix = mitk::ConstructScalingTransformation(parameters[6], parameters[7]);
  vtkSmartPointer<vtkMatrix4x4> output = vtkSmartPointer<vtkMatrix4x4>::New();
  mitk::CopyToVTK4x4Matrix(scalingMatrix, *output);
  return output;
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> UltrasoundPointerCalibrationCostFunction::GetScalingMatrix(const ParametersType & parameters) const
{
  cv::Matx44d rigidBodyMatrix = mitk::ConstructRodriguesTransformationMatrix(
        parameters[0], parameters[1], parameters[2],
        parameters[3], parameters[4], parameters[5]
      );
  vtkSmartPointer<vtkMatrix4x4> output = vtkSmartPointer<vtkMatrix4x4>::New();
  mitk::CopyToVTK4x4Matrix(rigidBodyMatrix, *output);
  return output;
}


//-----------------------------------------------------------------------------
UltrasoundPointerCalibrationCostFunction::MeasureType UltrasoundPointerCalibrationCostFunction::GetValue(const ParametersType & parameters) const
{
  this->ValidateSizeOfParametersArray(parameters);

  MeasureType value;
  value.SetSize(this->GetNumberOfValues());

  cv::Matx44d rigidBodyMatrix = mitk::ConstructRodriguesTransformationMatrix(
        parameters[0], parameters[1], parameters[2],
        parameters[3], parameters[4], parameters[5]
      );

  cv::Matx44d scalingMatrix = mitk::ConstructScalingTransformation(parameters[6], parameters[7]);

  cv::Matx44d calibrationMatrix = rigidBodyMatrix * scalingMatrix;

  mitk::PointSet::DataType* itkPointSet = m_ImagePoints->GetPointSet();
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType imagePointMITK;
  mitk::PointSet::PointType sensorPointMITK;

  cv::Matx41d imagePoint;
  cv::Matx41d transformedImagePoint;

  MeasureType::SizeValueType counter = 0;

  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();
    imagePointMITK = pIt->Value();

    // ALL points should exist, so throw error if not.
    if (!m_SensorPoints->GetPointIfExists(pointID, &sensorPointMITK))
    {
      mitkThrow() << "Failed to find sensor point " << pointID;
    }
    imagePoint(0,0) = imagePointMITK[0];
    imagePoint(1,0) = imagePointMITK[1];
    imagePoint(2,0) = imagePointMITK[2];
    imagePoint(3,0) = 1;

    // Transforms point from image to sensor.
    transformedImagePoint = calibrationMatrix * imagePoint;

    // Calculates the difference in x, y, z.
    value[counter] = transformedImagePoint(0,0) - sensorPointMITK[0];
    counter++;
    value[counter] = transformedImagePoint(1,0) - sensorPointMITK[1];
    counter++;
    value[counter] = transformedImagePoint(2,0) - sensorPointMITK[2];
    counter++;
  }
  //std::cerr << "UltrasoundPointerCalibrationCostFunction:" << parameters << "=" << this->GetResidual(value) << std::endl;
  return value;
}


//-----------------------------------------------------------------------------
} // end namespace
