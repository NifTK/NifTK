/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkUltrasoundPinCalibration_h
#define mitkUltrasoundPinCalibration_h

#include "niftkOpenCVExports.h"
#include "mitkInvariantPointCalibration.h"
#include "itkUltrasoundPinCalibrationCostFunction.h"

namespace mitk {

/**
 * \class UltrasoundPinCalibration
 * \brief Does an Ultrasound Pin/Cross-Wire calibration.
 */
class NIFTKOPENCV_EXPORT UltrasoundPinCalibration : public mitk::InvariantPointCalibration
{

public:

  mitkClassMacro(UltrasoundPinCalibration, mitk::InvariantPointCalibration);
  itkNewMacro(UltrasoundPinCalibration);

  void SetImageScaleFactors(const mitk::Point2D& point);
  mitk::Point2D GetImageScaleFactors() const;

  void SetOptimiseImageScaleFactors(const bool&);
  bool GetOptimiseImageScaleFactors() const;

  /**
   * \see mitk::InvariantPointCalibration::Calibrate().
   */
  virtual double Calibrate();

protected:

  UltrasoundPinCalibration();
  virtual ~UltrasoundPinCalibration();

  UltrasoundPinCalibration(const UltrasoundPinCalibration&); // Purposefully not implemented.
  UltrasoundPinCalibration& operator=(const UltrasoundPinCalibration&); // Purposefully not implemented.

private:

  itk::UltrasoundPinCalibrationCostFunction* m_DownCastCostFunction;

}; // end class

} // end namespace

#endif
