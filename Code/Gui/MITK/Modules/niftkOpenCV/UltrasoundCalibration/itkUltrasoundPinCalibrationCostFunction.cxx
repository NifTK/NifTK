/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkUltrasoundPinCalibrationCostFunction.h"
#include <sstream>
#include <mitkOpenCVMaths.h>

namespace itk {

//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::UltrasoundPinCalibrationCostFunction()
: m_NumberOfParameters(1)
, m_NumberOfValues(1)
{
  m_InvariantPoint.x = 0;
  m_InvariantPoint.y = 0;
  m_InvariantPoint.z = 0;
  m_MillimetresPerPixel.x = 1;
  m_MillimetresPerPixel.y = 1;
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::~UltrasoundPinCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
unsigned int UltrasoundPinCalibrationCostFunction::GetNumberOfParameters(void) const
{
  return m_NumberOfParameters;
}


//-----------------------------------------------------------------------------
unsigned int UltrasoundPinCalibrationCostFunction::GetNumberOfValues(void) const
{
  return m_NumberOfValues;
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetMatrices(const std::vector< cv::Mat >& matrices)
{
  m_Matrices = matrices;
  m_NumberOfValues = matrices.size();
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetPoints(const std::vector< cv::Point3d > points)
{
  m_Points = points;
  m_NumberOfValues = points.size();
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetNumberOfParameters(const int& numberOfParameters)
{
  m_NumberOfParameters = numberOfParameters;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetInvariantPoint(const cv::Point3d& invariantPoint)
{
  m_InvariantPoint = invariantPoint;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetMillimetresPerPixel(const cv::Point2d& mmPerPix)
{
  m_MillimetresPerPixel = mmPerPix;
  this->Modified();
}


//-----------------------------------------------------------------------------
double UltrasoundPinCalibrationCostFunction::GetResidual(const MeasureType & values) const
{
  double rmsError = 0;
  unsigned int numberOfValues = values.GetSize();

  if (numberOfValues > 0)
  {
    for (unsigned int i = 0; i < numberOfValues; i++)
    {
      rmsError += values[i];
    }

    rmsError /= (double)(numberOfValues);
    rmsError = sqrt(rmsError);
  }

  return rmsError;
}


//-----------------------------------------------------------------------------
cv::Matx44d UltrasoundPinCalibrationCostFunction::GetCalibrationTransformation(const ParametersType & parameters) const
{
  cv::Matx44d rigidTransformation = mitk::ConstructRodriguesTransformationMatrix(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]);
  return rigidTransformation;
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::MeasureType UltrasoundPinCalibrationCostFunction::GetValue(
  const ParametersType & parameters
  ) const
{
  if (parameters.GetSize() != m_NumberOfParameters)
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibrationCostFunction::GetValue given " << parameters.GetSize() << ", but was expecting " << m_NumberOfParameters << " parameters";
    throw std::logic_error(oss.str());
  }

  if (m_Matrices.size() == 0)
  {
    throw std::logic_error("UltrasoundPinCalibrationCostFunction::GetValue(): No matrices available");
  }

  if (m_Points.size() == 0)
  {
    throw std::logic_error("UltrasoundPinCalibrationCostFunction::GetValue():No points available");
  }

  if (m_Matrices.size() != m_Points.size())
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibrationCostFunction::GetValue(): The number of matrices (" << m_Matrices.size() << ") differs from the number of points (" << m_Points.size() << ")";
    throw std::logic_error(oss.str());
  }

  cv::Matx44d rigidTransformation = GetCalibrationTransformation(parameters);

  cv::Matx44d scalingTransformation;
  mitk::MakeIdentity(scalingTransformation);

  cv::Matx44d invariantPointTranslation;
  mitk::MakeIdentity(invariantPointTranslation);

  if (parameters.size() == 8 || parameters.size() == 11)
  {
    scalingTransformation(0, 0) = parameters[6];
    scalingTransformation(1, 1) = parameters[7];
  }
  else
  {
    // i.e. its not being optimised.
    scalingTransformation(0, 0) = m_MillimetresPerPixel.x;
    scalingTransformation(1, 1) = m_MillimetresPerPixel.y;
  }

  if (parameters.size() == 9 || parameters.size() == 11)
  {
    if (parameters.size() == 9)
    {
      invariantPointTranslation(0, 3) = parameters[6];
      invariantPointTranslation(1, 3) = parameters[7];
      invariantPointTranslation(2, 3) = parameters[8];
    }
    else
    {
      invariantPointTranslation(0, 3) = parameters[8];
      invariantPointTranslation(1, 3) = parameters[9];
      invariantPointTranslation(2, 3) = parameters[10];
    }
  }
  else
  {
    // i.e. its not being optimised.
    invariantPointTranslation(0, 3) = m_InvariantPoint.x;
    invariantPointTranslation(1, 3) = m_InvariantPoint.y;
    invariantPointTranslation(2, 3) = m_InvariantPoint.z;
  }

  MeasureType value;
  value.SetSize(m_NumberOfValues);

  for (unsigned int i = 0; i < m_Matrices.size(); i++)
  {
    cv::Matx44d trackerTransformation(m_Matrices[i]);
/*
    cv::Matx44d trackerTransformationFlipped = trackerTransformation;
    trackerTransformationFlipped(0,2) = trackerTransformation(0,2) * -1;
    trackerTransformationFlipped(1,2) = trackerTransformation(1,2) * -1;
    trackerTransformationFlipped(2,0) = trackerTransformation(2,0) * -1;
    trackerTransformationFlipped(2,1) = trackerTransformation(2,1) * -1;
    trackerTransformationFlipped(2,3) = trackerTransformation(2,3) * -1;
*/
    cv::Matx44d combinedTransformation = (trackerTransformation * (rigidTransformation * scalingTransformation));
    cv::Matx41d point, transformedPoint;

    point(0,0) = m_Points[i].x;
    point(1,0) = m_Points[i].y;
    point(2,0) = m_Points[i].z;
    point(3,0) = 1;

    transformedPoint = combinedTransformation * point;

    value[i] = (transformedPoint(0,0) - invariantPointTranslation(0,3)) * (transformedPoint(0,0) - invariantPointTranslation(0,3))
             + (transformedPoint(1,0) - invariantPointTranslation(1,3)) * (transformedPoint(1,0) - invariantPointTranslation(1,3))
             + (transformedPoint(2,0) - invariantPointTranslation(2,3)) * (transformedPoint(2,0) - invariantPointTranslation(2,3));
/*
    if (i == 0)
    {
      std::cerr << "Matt, p=" << point(0,0) << ", " << point(1,0) << ", " << point(2,0) << ", " << point(3,0) << std::endl;
      std::cerr << "Matt, tp=" << transformedPoint(0,0) << ", " << transformedPoint(1,0) << ", " << transformedPoint(2,0) << ", " << transformedPoint(3,0) << std::endl;
      std::cerr << "Matt, s=" << scalingTransformation(0,0) << ", " << scalingTransformation(0,1) << ", " << scalingTransformation(0,2) << ", " << scalingTransformation(0,3) << std::endl;
      std::cerr << "Matt, s=" << scalingTransformation(1,0) << ", " << scalingTransformation(1,1) << ", " << scalingTransformation(1,2) << ", " << scalingTransformation(1,3) << std::endl;
      std::cerr << "Matt, s=" << scalingTransformation(2,0) << ", " << scalingTransformation(2,1) << ", " << scalingTransformation(2,2) << ", " << scalingTransformation(2,3) << std::endl;
      std::cerr << "Matt, s=" << scalingTransformation(3,0) << ", " << scalingTransformation(3,1) << ", " << scalingTransformation(3,2) << ", " << scalingTransformation(3,3) << std::endl;
      std::cerr << "Matt, t=" << trackerTransformation(0,0) << ", " << trackerTransformation(0,1) << ", " << trackerTransformation(0,2) << ", " << trackerTransformation(0,3) << std::endl;
      std::cerr << "Matt, t=" << trackerTransformation(1,0) << ", " << trackerTransformation(1,1) << ", " << trackerTransformation(1,2) << ", " << trackerTransformation(1,3) << std::endl;
      std::cerr << "Matt, t=" << trackerTransformation(2,0) << ", " << trackerTransformation(2,1) << ", " << trackerTransformation(2,2) << ", " << trackerTransformation(2,3) << std::endl;
      std::cerr << "Matt, t=" << trackerTransformation(3,0) << ", " << trackerTransformation(3,1) << ", " << trackerTransformation(3,2) << ", " << trackerTransformation(3,3) << std::endl;
      std::cerr << "Matt, r=" << rigidTransformation(0,0) << ", " << rigidTransformation(0,1) << ", " << rigidTransformation(0,2) << ", " << rigidTransformation(0,3) << std::endl;
      std::cerr << "Matt, r=" << rigidTransformation(1,0) << ", " << rigidTransformation(1,1) << ", " << rigidTransformation(1,2) << ", " << rigidTransformation(1,3) << std::endl;
      std::cerr << "Matt, r=" << rigidTransformation(2,0) << ", " << rigidTransformation(2,1) << ", " << rigidTransformation(2,2) << ", " << rigidTransformation(2,3) << std::endl;
      std::cerr << "Matt, r=" << rigidTransformation(3,0) << ", " << rigidTransformation(3,1) << ", " << rigidTransformation(3,2) << ", " << rigidTransformation(3,3) << std::endl;

      std::cerr << "Matt, ip=" << invariantPointTranslation(0,3) << ", " << invariantPointTranslation(1,3) << ", " << invariantPointTranslation(2,3) << std::endl;
    }
*/
  }
  double rmsError = this->GetResidual(value);

  //std::cout << "UltrasoundPinCalibrationCostFunction::GetValue(" << parameters << ") = " << rmsError << std::endl;
  return value;
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::GetDerivative(
  const ParametersType & parameters,
  DerivativeType  & derivative
  ) const
{
  // Do forward differencing.
  MeasureType currentValue = this->GetValue(parameters);
  MeasureType forwardValue;

  ParametersType forwardParameters;
  derivative.SetSize(m_NumberOfParameters, m_NumberOfValues);

  ParametersType scales(m_NumberOfParameters);
  scales.Fill(1);

  if (parameters.size() == 8 || parameters.size() == 11)
  {
    scales[6] = 0.0001;
    scales[7] = 0.0001;
  }

  for (unsigned int i = 0; i < m_NumberOfParameters; i++)
  {
    forwardParameters = parameters;
    forwardParameters[i] += (1 * scales[i]);

    forwardValue = this->GetValue(forwardParameters);

    for (unsigned int j = 0; j < m_NumberOfValues; j++)
    {
      derivative[i][j] = forwardValue[j] - currentValue[j];
    }
  }
}

//-----------------------------------------------------------------------------
} // end namespace
