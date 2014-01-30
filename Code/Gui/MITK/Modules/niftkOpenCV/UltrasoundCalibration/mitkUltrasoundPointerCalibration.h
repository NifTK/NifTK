/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkUltrasoundPointerCalibration_h
#define mitkUltrasoundPointerCalibration_h

#include "niftkOpenCVExports.h"
#include "mitkUltrasoundCalibration.h"
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

namespace mitk {

/**
 * \class UltrasoundPointerCalibration
 * \brief Does an ultrasound probe calibration from an ordered list of tracker matrices,
 * and pin locations (x,y pixels) extracted from 2D ultrasound images.
 *
 * See: http://dx.doi.org/10.1016/S0301-5629(01)00469-0
 */
class NIFTKOPENCV_EXPORT UltrasoundPointerCalibration : public mitk::UltrasoundCalibration
{

public:

  mitkClassMacro(UltrasoundPointerCalibration, mitk::UltrasoundCalibration);
  itkNewMacro(UltrasoundPointerCalibration);

  itkSetMacro(PointerOffset, mitk::Point3D);
  itkGetMacro(PointerOffset, mitk::Point3D);

  void InitialisePointerOffset(const std::vector<float>& commandLineArgs);

  void InitialisePointerTrackerToProbeTrackerTransform(const std::string& fileName);

  void InitialiseProbeToProbeTrackerTransform(const std::string& fileName);

  /**
   * \brief Performs pin-head (invariant-point) calibration.
   * \see http://dx.doi.org/10.1016/S0301-5629(01)00469-0
   * \see mitk::UltrasoundCalibration::Calibrate()
   */
  virtual double Calibrate(
      const std::vector< cv::Mat >& matrices,
      const std::vector< cv::Point2d >& points,
      cv::Matx44d& outputMatrix
      );

protected:

  UltrasoundPointerCalibration();
  virtual ~UltrasoundPointerCalibration();

  UltrasoundPointerCalibration(const UltrasoundPointerCalibration&); // Purposefully not implemented.
  UltrasoundPointerCalibration& operator=(const UltrasoundPointerCalibration&); // Purposefully not implemented.

private:

  mitk::Point3D                 m_PointerOffset;
  vtkSmartPointer<vtkMatrix4x4> m_PointerTrackerToProbeTrackerTransform;
  vtkSmartPointer<vtkMatrix4x4> m_ProbeToProbeTrackerTransform;

}; // end class

} // end namespace

#endif
