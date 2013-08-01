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

#include <mitkVideoTrackerMatching.h>

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

  void InitialiseVideo ();
     
  itkSetMacro(FlipTracking, bool);
  itkSetMacro(FlipExtrinsic, bool);
  itkSetMacro(SortByDistance, bool);
  itkSetMacro(SortByAngle, bool);
  itkSetMacro(FramesToUse, unsigned int);
  itkSetMacro(Directory, std::string);
  itkSetMacro(TrackerIndex,int);
  itkSetMacro(AbsTrackerTimingError,long long);

  itkGetMacro(VideoInitialised, bool);
  itkGetMacro(TrackingDataInitialised, bool);
  
  itkSetMacro(NumberCornersWidth, unsigned int);
  itkSetMacro(NumberCornersHeight, unsigned int);
  itkSetMacro(SquareSizeInMillimetres, double);
  itkSetMacro(WriteOutChessboards,bool);

protected:

  HandeyeCalibrateFromDirectory();
  virtual ~HandeyeCalibrateFromDirectory();

  HandeyeCalibrateFromDirectory(const HandeyeCalibrateFromDirectory&); // Purposefully not implemented.
  HandeyeCalibrateFromDirectory& operator=(const HandeyeCalibrateFromDirectory&); // Purposefully not implemented.

private:
  bool                                m_FlipTracking;
  bool                                m_FlipExtrinsic;
  bool                                m_SortByDistance;
  bool                                m_SortByAngle;
  unsigned int                        m_FramesToUse; //target frames to use actual number used will depend on number of good frames
  float                               m_BadFrameFactor; //how many extra frames to stick in buffer
  bool                                m_SaveProcessedVideoData;

  bool                                m_VideoInitialised;
  bool                                m_TrackingDataInitialised;

  int                                 m_TrackerIndex;
  long long                           m_AbsTrackerTimingError;
  
  unsigned int                        m_NumberCornersWidth;
  unsigned int                        m_NumberCornersHeight;
  double                              m_SquareSizeInMillimetres;
  std::string                         m_Directory;
  bool                                m_WriteOutChessboards;

  mitk::VideoTrackerMatching::Pointer m_Matcher;

  // Not possible to store the frames in memory, will need to process them on the fly
  // First init videotrackermatching
  // use this to get the number of frames (stored in the framemap.log
  // Process frames, either sequencially of more likely by random selection
  std::vector <cv::Mat> m_LeftCameraVideoFrames;
  std::vector <cv::Mat> m_RightCameraVideoFrames;

  std::vector<std::string>      FindVideoData();
  void             LoadVideoData(std::string filename);

  /**
   * \brief As video processing can be time consuming allow for saving and loading 
   * of pre-processed video data
   */
  std::string      CheckForExistingData();




}; // end class

} // end namespace

#endif
