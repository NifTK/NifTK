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
#include "mitkUltrasoundCalibration.h"

namespace mitk {

/**
 * \class UltrasoundPinCalibration
 * \brief Does an ultrasound probe calibration from an ordered list of tracker matrices,
 * and pin locations (x,y pixels) extracted from 2D ultrasound images.
 */
class NIFTKOPENCV_EXPORT UltrasoundPinCalibration : public mitk::UltrasoundCalibration
{

public:

  mitkClassMacro(UltrasoundPinCalibration, mitk::UltrasoundCalibration);
  itkNewMacro(UltrasoundPinCalibration);

  itkSetMacro(InvariantPoint, mitk::Point3D);
  itkGetMacro(InvariantPoint, mitk::Point3D);

  itkSetMacro(OptimiseInvariantPoint, bool);
  itkGetMacro(OptimiseInvariantPoint, bool);

  void InitialiseInvariantPoint(const std::vector<float>& commandLineArgs);

  /**
   * \brief Performs pin-head (invariant-point) calibration.
   * \see mitk::UltrasoundCalibration::Calibrate()
   */
  virtual double Calibrate(
      const std::vector< cv::Mat >& matrices,
      const std::vector< std::pair<int, cv::Point2d> >& points,
      cv::Matx44d& outputMatrix
      );

protected:

  UltrasoundPinCalibration();
  virtual ~UltrasoundPinCalibration();

  UltrasoundPinCalibration(const UltrasoundPinCalibration&); // Purposefully not implemented.
  UltrasoundPinCalibration& operator=(const UltrasoundPinCalibration&); // Purposefully not implemented.

private:

  mitk::Point3D m_InvariantPoint;
  bool          m_OptimiseInvariantPoint;

}; // end class

} // end namespace

#endif
