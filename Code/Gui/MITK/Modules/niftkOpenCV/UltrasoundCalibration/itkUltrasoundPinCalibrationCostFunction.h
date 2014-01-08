/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUltrasoundPinCalibrationCostFunction_h
#define itkUltrasoundPinCalibrationCostFunction_h

#include <itkMultipleValuedCostFunction.h>
#include <cv.h>

namespace itk {

/**
 * \class UltrasoundPinCalibrationCostFunction
 * \brief Multi-valued cost function adaptor, to plug into Levenberg-Marquardt,
 * minimising the squared distance error of a cloud of points from the invariant point (normally 0,0,0).
 *
 * The parameters array should be set before optimisation with a reasonable starting estimate
 * using the this->SetInitialPosition(parameters) method in the base class. This class
 * can optimise different numbers of degrees of freedom as follows:
 * <pre>
 * 6DOF: 6 rigid (rx, ry, rz in radians, tx, ty, tz in millimetres).
 * 7DOF: 6 rigid + 1 scaling
 * 9DOF: 6 rigid + 3 invariant point (x, y, z location).
 * 10DOF: 6 rigid + 1 scaling + 3 invariant point.
 * </pre>
 * The order of parameters is important.
 */
class UltrasoundPinCalibrationCostFunction : public itk::MultipleValuedCostFunction
{

public:

  typedef UltrasoundPinCalibrationCostFunction Self;
  typedef itk::MultipleValuedCostFunction      Superclass;
  typedef itk::SmartPointer<Self>              Pointer;
  typedef itk::SmartPointer<const Self>        ConstPointer;

  itkNewMacro( Self );

  typedef Superclass::ParametersType           ParametersType;
  typedef Superclass::DerivativeType           DerivativeType;
  typedef Superclass::MeasureType              MeasureType;

  /**
   * \brief Sets (copies) the matrices into this object, and sets the number of values accordingly.
   */
  void SetMatrices(const std::vector< cv::Mat >& matrices);

  /**
   * \brief Sets (copies) the point data into this object, and sets the number of values accordingly.
   */
  void SetPoints(const std::vector< cv::Point3d > points);

  /**
   * \brief Sets the number of parameters being optimised.
   */
  void SetNumberOfParameters(const int& numberOfParameters);

  /**
   * \brief Sets the initial invariant point.
   */
  void SetInvariantPoint(const cv::Point3d& invariantPoint);

  /**
   * \brief Sets the initial millimetres per pixel.
   */
  void SetMillimetresPerPixel(const double& mmPerPix);

  /**
   * \brief Required by base class to return the number of parameters, where
   * here we have 6 (rigid), 6 (rigid) +1 (scaling), 6 (rigid) +3 (invariant point)
   * or 6 (rigid) + 1 (scaling) + 3 (invariant point).
   */
  virtual unsigned int GetNumberOfParameters() const;

  /**
   * \brief Equal to the number of points * 3.
   */
  virtual unsigned int GetNumberOfValues(void) const;

  /**
   * \brief The cost function is the residual error of the reconstructed point,
   * where this function returns an array of n (x,y,z) tuples where n is the number
   * of points, and each x,y,z measure is the distance from zero in that axis.
   *
   * So the cost function is calculated by taking each point, transforming into phantom space,
   * and measuring the squared distance to the origin. i.e. its the size of the reconstructed point cloud.
   */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /**
   * \brief Simply uses forward differences to approximate the derivative for each of the parameters.
   */
  virtual void GetDerivative( const ParametersType & parameters, DerivativeType  & derivative ) const;

  /**
   * \brief Computes the 6DOF transformation from image to sensor, (i.e. without scaling parameters).
   */
  cv::Matx44d GetCalibrationTransformation(const ParametersType & parameters) const;

  /**
   * \brief Returns the residual.
   */
  double GetResidual(const MeasureType& values) const;

  /**
   * \brief Used when calculating derivative using forward differences.
   */
  void SetScales(const ParametersType& scales);

  /**
   * \brief Sets the initial calibration matrix.
   */
  void SetInitialGuess(const cv::Matx44d& initialGuess);

protected:

  UltrasoundPinCalibrationCostFunction();
  virtual ~UltrasoundPinCalibrationCostFunction();

  UltrasoundPinCalibrationCostFunction(const UltrasoundPinCalibrationCostFunction&); // Purposefully not implemented.
  UltrasoundPinCalibrationCostFunction& operator=(const UltrasoundPinCalibrationCostFunction&); // Purposefully not implemented.

private:

  std::vector< cv::Mat >     m_Matrices;
  std::vector< cv::Point3d > m_Points;
  cv::Point3d                m_InvariantPoint;
  double                     m_MillimetresPerPixel;
  unsigned int               m_NumberOfParameters;
  mutable unsigned int       m_NumberOfValues;
  ParametersType             m_Scales;
  cv::Matx44d                m_InitialGuess;
};

} // end namespace

#endif
