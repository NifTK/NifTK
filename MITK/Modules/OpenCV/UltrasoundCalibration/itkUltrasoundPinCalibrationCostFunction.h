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

#include <itkInvariantPointCalibrationCostFunction.h>

namespace itk {

/**
 * \class UltrasoundPinCalibrationCostFunction
 * \brief Minimises the RMS error around a stationary invariant point.
 * \see itk::InvariantPointCalibrationCostFunction
 * \see itk::VideoHandEyeCalibrationCostFunction
 */
class UltrasoundPinCalibrationCostFunction : public itk::InvariantPointCalibrationCostFunction
{

public:

  typedef UltrasoundPinCalibrationCostFunction       Self;
  typedef itk::InvariantPointCalibrationCostFunction Superclass;
  typedef itk::SmartPointer<Self>                    Pointer;
  typedef itk::SmartPointer<const Self>              ConstPointer;

  itkNewMacro( Self );

  typedef Superclass::ParametersType                 ParametersType;
  typedef Superclass::DerivativeType                 DerivativeType;
  typedef Superclass::MeasureType                    MeasureType;

  itkSetMacro(ScaleFactors, mitk::Point2D);
  itkGetConstMacro(ScaleFactors, mitk::Point2D);

  itkSetMacro(OptimiseScaleFactors, bool);
  itkGetConstMacro(OptimiseScaleFactors, bool);

  /**
   * \see itk::InvariantPointCalibrationCostFunction::GetCalibrationTransformation().
   */
  virtual cv::Matx44d GetCalibrationTransformation(const ParametersType & parameters) const;

protected:

  UltrasoundPinCalibrationCostFunction();
  virtual ~UltrasoundPinCalibrationCostFunction();

  UltrasoundPinCalibrationCostFunction(const UltrasoundPinCalibrationCostFunction&); // Purposefully not implemented.
  UltrasoundPinCalibrationCostFunction& operator=(const UltrasoundPinCalibrationCostFunction&); // Purposefully not implemented.

  /**
   * \brief Computes the scaling transformation.
   */
  cv::Matx44d GetScalingTransformation(const ParametersType & parameters) const;

private:

  mitk::Point2D m_ScaleFactors;
  bool          m_OptimiseScaleFactors;

};

} // end namespace

#endif
