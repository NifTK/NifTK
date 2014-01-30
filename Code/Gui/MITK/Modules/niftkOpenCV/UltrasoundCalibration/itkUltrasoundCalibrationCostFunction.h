/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUltrasoundCalibrationCostFunction_h
#define itkUltrasoundCalibrationCostFunction_h

#include <itkMultipleValuedCostFunction.h>
#include <cv.h>
#include <mitkVector.h>

namespace itk {

/**
 * \class UltrasoundCalibrationCostFunction
 * \brief Base class for itk::UltrasoundPinCalibrationCostFunction and itk::UltrasoundPointerCalibrationCostFunction,
 * creating a multi-value cost function to plug into a Levenberg-Marquardt optimiser.
 */
class UltrasoundCalibrationCostFunction : public itk::MultipleValuedCostFunction
{

public:

  typedef UltrasoundCalibrationCostFunction    Self;
  typedef itk::MultipleValuedCostFunction      Superclass;
  typedef itk::SmartPointer<Self>              Pointer;
  typedef itk::SmartPointer<const Self>        ConstPointer;

  typedef Superclass::ParametersType           ParametersType;
  typedef Superclass::DerivativeType           DerivativeType;
  typedef Superclass::MeasureType              MeasureType;

  /**
   * \brief Equal to the number of points * 3.
   */
  virtual unsigned int GetNumberOfValues(void) const;

  /**
   * \brief Required by base class to return the number of parameters.
   */
  virtual unsigned int GetNumberOfParameters() const;

  /**
   * \brief Sets the number of parameters being optimised.
   */
  void SetNumberOfParameters(const int& numberOfParameters);

  /**
   * \brief Stores an initial 6DOF calibration transformation.
   * \param initialGuess vector of 3 Rodrigues rotations, followed by 3 translation parameters in millimetres.
   */
  void SetInitialGuess(const std::vector<double>& initialGuess);

  /**
   * \brief Sets (copies) the tracking matrices into this object, and sets the number of values accordingly.
   */
  void SetMatrices(const std::vector< cv::Mat >& matrices);

  /**
   * \brief Sets (copies) the 2D ultrasound point data into this object.
   */
  void SetPoints(const std::vector< cv::Point2d > points);

  /**
   * \brief Sets the scale factors for the ultrasound image in millimetres per pixel.
   */
  void SetMillimetresPerPixel(const mitk::Point2D& mmPerPix);

  /**
   * \brief Used when calculating derivative using forward differences.
   */
  void SetScales(const ParametersType& scales);

  /**
   * \brief Computes the 6DOF transformation from image to sensor, (i.e. without scaling parameters).
   */
  cv::Matx44d GetCalibrationTransformation(const ParametersType & parameters) const;

  /**
   * \brief Returns the RMS residual.
   */
  double GetResidual(const MeasureType& values) const;

  /**
   * \brief Simply uses forward differences to approximate the derivative for each of the parameters.
   */
  virtual void GetDerivative( const ParametersType & parameters, DerivativeType  & derivative ) const;

protected:

  UltrasoundCalibrationCostFunction();
  virtual ~UltrasoundCalibrationCostFunction();

  UltrasoundCalibrationCostFunction(const UltrasoundCalibrationCostFunction&); // Purposefully not implemented.
  UltrasoundCalibrationCostFunction& operator=(const UltrasoundCalibrationCostFunction&); // Purposefully not implemented.

  void ValidateSizeOfParametersArray(const ParametersType & parameters) const;
  void ValidateSizeOfScalesArray(const ParametersType & parameters) const;

  std::vector<double>        m_InitialGuess;
  std::vector< cv::Mat >     m_Matrices;
  std::vector< cv::Point2d > m_Points;
  bool                       m_OptimiseScaling;
  mitk::Point2D              m_MillimetresPerPixel;
  ParametersType             m_Scales;
  mutable unsigned int       m_NumberOfValues;
  unsigned int               m_NumberOfParameters;
};

} // end namespace

#endif
