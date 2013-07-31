/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkHandeyeCalibrateFromDirectory_h
#define mitkHandeyeCalibrate_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <cv.h>

namespace mitk {

/**
 * \class HandeyeCalibrateFromDirectory_h
 * \brief Computes the handeye calibration using extrinsic calibration and tracking
 * data, using TSAI's least squares method. The class searches the directory for 
 * a suitable video file, frame map and tracking data. 
 */
class NIFTKOPENCV_EXPORT HandeyeCalibrateFromDirectory : public itk::Object
{

public:

  mitkClassMacro(HandeyeCalibrateFromDirectory, itk::Object);
  itkNewMacro(HandeyeCalibrateFromDirectory);
  
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
  itkSetMacro(FramesToUse, int);

  itkGetMacro(VideoInitialised, bool);
  itkGetMacro(TrackingDataInitialised, bool);

protected:

  HandeyeCalibrateFromDirectory();
  virtual ~HandeyeCalibrateFromDirectory();

  HandeyeCalibrateFromDirectory(const HandeyeCalibrateFromDirectory&); // Purposefully not implemented.
  HandeyeCalibrateFromDirectory& operator=(const HandeyeCalibrateFromDirectory&); // Purposefully not implemented.

private:
  bool                  m_FlipTracking;
  bool                  m_FlipExtrinsic;
  bool                  m_SortByDistance;
  bool                  m_SortByAngle;
  int                   m_FramesToUse;

  bool                  m_VideoInitialised;
  bool                  m_TrackingDataInitialised;

  std::string           m_Directory;

  // lets store the video data it'll chew up memory but will probably be the most efficient 
  // method in most cases
  std::vector <cv::Mat> m_LeftCameraVideoFrames;
  std::vector <cv::Mat> m_RightCameraVideoFrames;

  std::string      FindVideoData();
  void      LoadVideoData(std::string filename);

}; // end class

} // end namespace

#endif
