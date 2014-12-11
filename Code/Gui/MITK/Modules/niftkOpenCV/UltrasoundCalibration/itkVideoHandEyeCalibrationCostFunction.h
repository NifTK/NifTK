/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkVideoHandEyeCalibrationCostFunction_h
#define itkVideoHandEyeCalibrationCostFunction_h

#include <itkInvariantPointCalibrationCostFunction.h>

namespace itk {

/**
 * \class VideoHandEyeCalibrationCostFunction
 * \brief Minimises the RMS error around a stationary invariant point.
 * \see itk::InvariantPointCalibrationCostFunction
 * \see itk::UltrasoundPinCalibrationCostFunction
 */
class VideoHandEyeCalibrationCostFunction : public itk::InvariantPointCalibrationCostFunction
{

public:

  typedef VideoHandEyeCalibrationCostFunction        Self;
  typedef itk::InvariantPointCalibrationCostFunction Superclass;
  typedef itk::SmartPointer<Self>                    Pointer;
  typedef itk::SmartPointer<const Self>              ConstPointer;

  itkNewMacro( Self );

  typedef Superclass::ParametersType                 ParametersType;
  typedef Superclass::DerivativeType                 DerivativeType;
  typedef Superclass::MeasureType                    MeasureType;

  /**
   * \see itk::InvariantPointCalibrationCostFunction::GetCalibrationTransformation().
   */
  virtual cv::Matx44d GetCalibrationTransformation(const ParametersType & parameters) const;

protected:

  VideoHandEyeCalibrationCostFunction();
  virtual ~VideoHandEyeCalibrationCostFunction();

  VideoHandEyeCalibrationCostFunction(const VideoHandEyeCalibrationCostFunction&); // Purposefully not implemented.
  VideoHandEyeCalibrationCostFunction& operator=(const VideoHandEyeCalibrationCostFunction&); // Purposefully not implemented.

private:

};

} // end namespace

#endif
