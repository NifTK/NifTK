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
 * \brief Multi-valued cost function adaptor, to plug into Levenberg-Marquardt.
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

  void SetMatrices(const std::vector< cv::Mat >& matrices);
  void SetPoints(const std::vector< cv::Point2d > points);
  void SetNumberOfParameters(const int& numberOfParameters);
  void SetInvariantPoint(const cv::Point3d& invariantPoint);
  void SetMillimetresPerPixel(const cv::Point2d& mmPerPix);

  virtual unsigned int GetNumberOfParameters() const;
  virtual unsigned int GetNumberOfValues(void) const;

  /**
   * \brief The cost function is the residual error of the reconstructed point.
   *
   * The 'invariant point' is assumed to be at the origin of its own coordinate system.
   *
   * So the cost function is calculated by taking each point, transforming into phantom space,
   * and measuring the squared distance to the origin. i.e. its the size of the reconstructed point cloud.
   */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /**
   * \brief
   */
  virtual void GetDerivative( const ParametersType & parameters, DerivativeType  & derivative ) const;

protected:

  UltrasoundPinCalibrationCostFunction();
  virtual ~UltrasoundPinCalibrationCostFunction();

  UltrasoundPinCalibrationCostFunction(const UltrasoundPinCalibrationCostFunction&); // Purposefully not implemented.
  UltrasoundPinCalibrationCostFunction& operator=(const UltrasoundPinCalibrationCostFunction&); // Purposefully not implemented.

private:

  std::vector< cv::Mat >     m_Matrices;
  std::vector< cv::Point2d > m_Points;
  unsigned int               m_NumberOfParameters;
  unsigned int               m_NumberOfValues;
  cv::Point3d                m_InvariantPoint;
  cv::Point2d                m_MillimetresPerPixel;
};

} // end namespace

#endif
