/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkInvariantPointCalibration_h
#define mitkInvariantPointCalibration_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>
#include <cv.h>
#include <mitkTimeStampsContainer.h>
#include <mitkTrackingAndTimeStampsContainer.h>
#include <itkInvariantPointCalibrationCostFunction.h>

namespace mitk {

/**
 * \class InvariantPointCalibration
 * \brief Base class for Ultrasound Pin/Cross-Wire calibration and Video Hand-Eye calibration.
 */
class NIFTKOPENCV_EXPORT InvariantPointCalibration : public itk::Object
{

public:

  mitkClassMacro(InvariantPointCalibration, itk::Object);

  typedef mitk::TimeStampsContainer::TimeStamp TimeStampType;

  void SetInvariantPoint(const mitk::Point3D& point);
  mitk::Point3D GetInvariantPoint() const;

  void SetOptimiseInvariantPoint(const bool&);
  bool GetOptimiseInvariantPoint() const;

  void SetTimingLag(const TimeStampType& timeStamp);
  TimeStampType GetTimingLag();

  void SetOptimiseTimingLag(const bool&);
  bool GetOptimiseTimingLag() const;

  void SetRigidTransformation(const cv::Matx44d& rigidBodyTrans);
  cv::Matx44d GetRigidTransformation() const;

  /**
   * \brief Loads a 4x4 matrix for the initial guess of the rigid part of the transformation.
   */
  void LoadRigidTransformation(const std::string& fileName);

  /**
   * \brief Saves the 4x4 matrix (after calibration).
   */
  void SaveRigidTransformation(const std::string& fileName);

  /**
   * \brief Sets the tracking data onto this object.
   */
  void SetTrackingData(mitk::TrackingAndTimeStampsContainer* trackingData);

  /**
   * \brief Sets the point data onto this object.
   */
  void SetPointData(std::vector< std::pair<unsigned long long, cv::Point3d> >* pointData);

  /**
   * \brief Derived classes implement the calibration method.
   */
  virtual double Calibrate() = 0;

protected:

  InvariantPointCalibration();
  virtual ~InvariantPointCalibration();

  InvariantPointCalibration(const InvariantPointCalibration&); // Purposefully not implemented.
  InvariantPointCalibration& operator=(const InvariantPointCalibration&); // Purposefully not implemented.

protected:

  itk::InvariantPointCalibrationCostFunction::Pointer        m_CostFunction; // constructor in derived classes MUST create one.
  std::vector< std::pair<unsigned long long, cv::Point3d> > *m_PointData;
  mitk::TrackingAndTimeStampsContainer                      *m_TrackingData;
  std::vector<double>                                        m_RigidTransformation;

}; // end class

} // end namespace

#endif
