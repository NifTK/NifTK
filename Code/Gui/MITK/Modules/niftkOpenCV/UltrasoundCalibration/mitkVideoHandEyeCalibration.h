/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkVideoHandEyeCalibration_h
#define mitkVideoHandEyeCalibration_h

#include "niftkOpenCVExports.h"
#include "mitkInvariantPointCalibration.h"
#include "itkVideoHandEyeCalibrationCostFunction.h"

namespace mitk {

/**
 * \class VideoHandEyeCalibration
 * \brief Does an Ultrasound Pin/Cross-Wire calibration.
 */
class NIFTKOPENCV_EXPORT VideoHandEyeCalibration : public mitk::InvariantPointCalibration
{

public:

  mitkClassMacro(VideoHandEyeCalibration, mitk::InvariantPointCalibration);
  itkNewMacro(VideoHandEyeCalibration);

  /**
   * \see mitk::InvariantPointCalibration::Calibrate().
   */
  virtual double Calibrate();

protected:

  VideoHandEyeCalibration();
  virtual ~VideoHandEyeCalibration();

  VideoHandEyeCalibration(const VideoHandEyeCalibration&); // Purposefully not implemented.
  VideoHandEyeCalibration& operator=(const VideoHandEyeCalibration&); // Purposefully not implemented.

private:

  double DoCalibration();

  itk::VideoHandEyeCalibrationCostFunction* m_DownCastCostFunction;

}; // end class

} // end namespace

#endif
