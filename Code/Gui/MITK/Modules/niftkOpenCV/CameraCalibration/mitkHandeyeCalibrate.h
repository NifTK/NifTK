/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkHandeyeCalibrate_h
#define mitkHandeyeCalibrate_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <cv.h>

namespace mitk {

/**
 * \class HandeyeCalibrate_h
 * \brief Computes the handeye calibration using extrinsic calibration and tracking
 * data, using TSAI's least squares method
 */
class NIFTKOPENCV_EXPORT HandeyeCalibrate : public itk::Object
{

public:

  mitkClassMacro(HandeyeCalibrate, itk::Object);
  itkNewMacro(HandeyeCalibrate);
  
  /**
   * \brief Calibration function that returns the residual errors, rotation and 
   * translational. If a ground truth solution is passed it returns a vector of 
   * differences for testing.
   */
  std::vector<double> Calibrate (const std::string& TrackingFileDirectory,
      const std::string& ExtrinsicFileDirectoryOrFile,
      const std::string GroundTruthSolution = "");

  itkSetMacro(FlipTracking, bool);
  itkSetMacro(FlipExtrinsic, bool);
  itkSetMacro(SortByDistance, bool);
  itkSetMacro(SortByAngle, bool);
  itkSetMacro(DoGridToWorld, bool);
  itkGetMacro(CameraToMarker, cv::Mat);
  itkGetMacro(GridToWorld, cv::Mat);

protected:

  HandeyeCalibrate();
  virtual ~HandeyeCalibrate();

  HandeyeCalibrate(const HandeyeCalibrate&); // Purposefully not implemented.
  HandeyeCalibrate& operator=(const HandeyeCalibrate&); // Purposefully not implemented.
  
  bool  m_FlipTracking;
  bool  m_FlipExtrinsic;
  bool  m_SortByDistance;
  bool  m_SortByAngle;
  bool  m_DoGridToWorld;

private: 

  cv::Mat m_CameraToMarker; //the handeye matrix
  cv::Mat m_GridToWorld;    //the position of the calibration grid in world coordinates
}; // end class

} // end namespace

#endif
