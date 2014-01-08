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
  mitk::MakeIdentity(m_InitialGuess);
  m_InvariantPoint.x = 0;
  m_InvariantPoint.y = 0;
  m_InvariantPoint.z = 0;
  m_MillimetresPerPixel = 1;
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
  m_NumberOfValues = matrices.size() * 3;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetPoints(const std::vector< cv::Point3d > points)
{
  m_Points = points;
  m_NumberOfValues = points.size() * 3;
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
void UltrasoundPinCalibrationCostFunction::SetMillimetresPerPixel(const double& mmPerPix)
{
  m_MillimetresPerPixel = mmPerPix;
  this->Modified();
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetInitialGuess(const cv::Matx44d& initialGuess)
{
  m_InitialGuess = initialGuess;
  this->Modified();
}


//-----------------------------------------------------------------------------
double UltrasoundPinCalibrationCostFunction::GetResidual(const MeasureType& values) const
{
  double residual = 0;

  for (unsigned int i = 0; i < values.size(); i++)
  {
    residual += fabs(static_cast<double>(values[i]));
  }

  return residual;
}


//-----------------------------------------------------------------------------
cv::Matx44d UltrasoundPinCalibrationCostFunction::GetCalibrationTransformation(const ParametersType & parameters) const
{
  double alpha = parameters[0];
  double beta = parameters[1];
  double gamma = parameters[2];

  cv::Matx44d rigidTransformation;
  mitk::MakeIdentity(rigidTransformation);

  rigidTransformation(0,0) = cos(beta)*cos(gamma);
  rigidTransformation(0,1) = cos(gamma)*sin(beta)*sin(alpha)+sin(gamma)*cos(alpha);
  rigidTransformation(0,2) = sin(alpha)*sin(gamma) - cos(gamma)*sin(beta)*cos(alpha);
  rigidTransformation(0,3) = parameters[3];
  rigidTransformation(1,0) = -sin(gamma)*cos(beta);
  rigidTransformation(1,1) = cos(gamma)*cos(alpha) - sin(gamma)*sin(beta)*sin(alpha);
  rigidTransformation(1,2) = sin(beta)*cos(alpha)*sin(gamma)+cos(gamma)*sin(alpha);
  rigidTransformation(1,3) = parameters[4];
  rigidTransformation(2,0) = sin(beta);
  rigidTransformation(2,1) = -sin(alpha)*cos(beta);
  rigidTransformation(2,2) = cos(alpha)*cos(beta);
  rigidTransformation(2,3) = parameters[5];

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

  if (parameters.size() == 7 || parameters.size() == 10)
  {
    double actualScale = m_MillimetresPerPixel * parameters[6];
    scalingTransformation(0, 0) = actualScale;
    scalingTransformation(1, 1) = actualScale;
  }
  else
  {
    // i.e. its not being optimised.
    scalingTransformation(0, 0) = m_MillimetresPerPixel;
    scalingTransformation(1, 1) = m_MillimetresPerPixel;
  }

  if (parameters.size() == 9 || parameters.size() == 10)
  {
    if (parameters.size() == 9)
    {
      invariantPointTranslation(0, 3) = -parameters[6];
      invariantPointTranslation(1, 3) = -parameters[7];
      invariantPointTranslation(2, 3) = -parameters[8];
    }
    else
    {
      invariantPointTranslation(0, 3) = -parameters[7];
      invariantPointTranslation(1, 3) = -parameters[8];
      invariantPointTranslation(2, 3) = -parameters[9];
    }
  }
  else
  {
    // i.e. its not being optimised.
    invariantPointTranslation(0, 3) = -m_InvariantPoint.x;
    invariantPointTranslation(1, 3) = -m_InvariantPoint.y;
    invariantPointTranslation(2, 3) = -m_InvariantPoint.z;
  }

  MeasureType value;
  value.SetSize(m_NumberOfValues);

  for (unsigned int i = 0; i < m_Matrices.size(); i++)
  {
    cv::Matx44d trackerTransformation(m_Matrices[i]);
    cv::Matx44d combinedTransformation = invariantPointTranslation * (trackerTransformation * (rigidTransformation * (m_InitialGuess * scalingTransformation)));
    cv::Matx41d point, transformedPoint;

    point(0,0) = m_Points[i].x;
    point(1,0) = m_Points[i].y;
    point(2,0) = m_Points[i].z;
    point(3,0) = 1;

    transformedPoint = combinedTransformation * point;

    value[i*3 + 0] = transformedPoint(0, 0);
    value[i*3 + 1] = transformedPoint(1, 0);
    value[i*3 + 2] = transformedPoint(2, 0);
/*
    if (i==-1)
    {
      std::cerr << "Matt, s=" << scalingTransformation(0,0) << ", " << scalingTransformation(0, 1) << ", " << scalingTransformation(0, 2) << ", " << scalingTransformation(0,3) << std::endl;
      std::cerr << "Matt, s=" << scalingTransformation(1,0) << ", " << scalingTransformation(1, 1) << ", " << scalingTransformation(1, 2) << ", " << scalingTransformation(1,3) << std::endl;
      std::cerr << "Matt, s=" << scalingTransformation(2,0) << ", " << scalingTransformation(2, 1) << ", " << scalingTransformation(2, 2) << ", " << scalingTransformation(2,3) << std::endl;
      std::cerr << "Matt, s=" << scalingTransformation(3,0) << ", " << scalingTransformation(3, 1) << ", " << scalingTransformation(3, 2) << ", " << scalingTransformation(3,3) << std::endl;

      std::cerr << "Matt, i=" << m_InitialGuess(0,0) << ", " << m_InitialGuess(0, 1) << ", " << m_InitialGuess(0, 2) << ", " << m_InitialGuess(0,3) << std::endl;
      std::cerr << "Matt, i=" << m_InitialGuess(1,0) << ", " << m_InitialGuess(1, 1) << ", " << m_InitialGuess(1, 2) << ", " << m_InitialGuess(1,3) << std::endl;
      std::cerr << "Matt, i=" << m_InitialGuess(2,0) << ", " << m_InitialGuess(2, 1) << ", " << m_InitialGuess(2, 2) << ", " << m_InitialGuess(2,3) << std::endl;
      std::cerr << "Matt, i=" << m_InitialGuess(3,0) << ", " << m_InitialGuess(3, 1) << ", " << m_InitialGuess(3, 2) << ", " << m_InitialGuess(3,3) << std::endl;

      std::cerr << "Matt, r=" << rigidTransformation(0,0) << ", " << rigidTransformation(0, 1) << ", " << rigidTransformation(0, 2) << ", " << rigidTransformation(0,3) << std::endl;
      std::cerr << "Matt, r=" << rigidTransformation(1,0) << ", " << rigidTransformation(1, 1) << ", " << rigidTransformation(1, 2) << ", " << rigidTransformation(1,3) << std::endl;
      std::cerr << "Matt, r=" << rigidTransformation(2,0) << ", " << rigidTransformation(2, 1) << ", " << rigidTransformation(2, 2) << ", " << rigidTransformation(2,3) << std::endl;
      std::cerr << "Matt, r=" << rigidTransformation(3,0) << ", " << rigidTransformation(3, 1) << ", " << rigidTransformation(3, 2) << ", " << rigidTransformation(3,3) << std::endl;

      std::cerr << "Matt, t=" << trackerTransformation(0,0) << ", " << trackerTransformation(0, 1) << ", " << trackerTransformation(0, 2) << ", " << trackerTransformation(0,3) << std::endl;
      std::cerr << "Matt, t=" << trackerTransformation(1,0) << ", " << trackerTransformation(1, 1) << ", " << trackerTransformation(1, 2) << ", " << trackerTransformation(1,3) << std::endl;
      std::cerr << "Matt, t=" << trackerTransformation(2,0) << ", " << trackerTransformation(2, 1) << ", " << trackerTransformation(2, 2) << ", " << trackerTransformation(2,3) << std::endl;
      std::cerr << "Matt, t=" << trackerTransformation(3,0) << ", " << trackerTransformation(3, 1) << ", " << trackerTransformation(3, 2) << ", " << trackerTransformation(3,3) << std::endl;

      std::cerr << "Matt, ipt=" << invariantPointTranslation(0,0) << ", " << invariantPointTranslation(0,1) << ", " << invariantPointTranslation(0,2) << ", " << invariantPointTranslation(0,3) << std::endl;
      std::cerr << "Matt, ipt=" << invariantPointTranslation(1,0) << ", " << invariantPointTranslation(1,1) << ", " << invariantPointTranslation(1,2) << ", " << invariantPointTranslation(1,3) << std::endl;
      std::cerr << "Matt, ipt=" << invariantPointTranslation(2,0) << ", " << invariantPointTranslation(2,1) << ", " << invariantPointTranslation(2,2) << ", " << invariantPointTranslation(2,3) << std::endl;
      std::cerr << "Matt, ipt=" << invariantPointTranslation(3,0) << ", " << invariantPointTranslation(3,1) << ", " << invariantPointTranslation(3,2) << ", " << invariantPointTranslation(3,3) << std::endl;

      std::cerr << "Matt, ct=" << combinedTransformation(0,0) << ", " << combinedTransformation(0,1) << ", " << combinedTransformation(0,2) << ", " << combinedTransformation(0,3) << std::endl;
      std::cerr << "Matt, ct=" << combinedTransformation(1,0) << ", " << combinedTransformation(1,1) << ", " << combinedTransformation(1,2) << ", " << combinedTransformation(1,3) << std::endl;
      std::cerr << "Matt, ct=" << combinedTransformation(2,0) << ", " << combinedTransformation(2,1) << ", " << combinedTransformation(2,2) << ", " << combinedTransformation(2,3) << std::endl;
      std::cerr << "Matt, ct=" << combinedTransformation(3,0) << ", " << combinedTransformation(3,1) << ", " << combinedTransformation(3,2) << ", " << combinedTransformation(3,3) << std::endl;

      std::cerr << "Matt, p=" << point(0,0) << ", " << point(1,0) << ", " << point(2,0) << std::endl;
      std::cerr << "Matt, tp=" << transformedPoint(0,0) << ", " << transformedPoint(1,0) << ", " << transformedPoint(2,0) << std::endl;
      std::cerr << "Matt, ip=" << invariantPointTranslation(0,3) << ", " << invariantPointTranslation(1,3) << ", " << invariantPointTranslation(2,3) << std::endl;
    }
*/
  }

  double residual = this->GetResidual(value);
  std::cout << "UltrasoundPinCalibrationCostFunction::GetValue(" << parameters << ") = " << residual << std::endl;

  return value;
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::SetScales(const ParametersType& scales)
{
  m_Scales = scales;
}


//-----------------------------------------------------------------------------
void UltrasoundPinCalibrationCostFunction::GetDerivative(
  const ParametersType & parameters,
  DerivativeType  & derivative
  ) const
{
  if (parameters.GetSize() != m_Scales.GetSize())
  {
    std::ostringstream oss;
    oss << "UltrasoundPinCalibrationCostFunction::GetDerivative given " << parameters.GetSize() << " parameters, but the scale factors array has " << m_Scales.GetSize() << " parameters";
    throw std::logic_error(oss.str());
  }

  // Do forward differencing.
  MeasureType currentValue = this->GetValue(parameters);
  MeasureType forwardValue;

  ParametersType forwardParameters;
  derivative.SetSize(m_NumberOfParameters, m_NumberOfValues);

  for (unsigned int i = 0; i < m_NumberOfParameters; i++)
  {
    forwardParameters = parameters;
    forwardParameters[i] += m_Scales[i];

    forwardValue = this->GetValue(forwardParameters);

    for (unsigned int j = 0; j < m_NumberOfValues; j++)
    {
      derivative[i][j] = forwardValue[j] - currentValue[j];
    }
  }
}

//-----------------------------------------------------------------------------
} // end namespace
