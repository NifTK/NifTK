/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkHandeyeCalibrateFromDirectory_h
#define mitkHandeyeCalibrateFromDirectory_h

#include "niftkOpenCVExports.h"
#include "mitkHandeyeCalibrate.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>
#include <cv.h>

#include <mitkVideoTrackerMatching.h>

namespace mitk {

/**
 * \class HandeyeCalibrateFromDirectory_h
 * \brief Computes the handeye calibration using extrinsic calibration and tracking
 * data, using TSAI's least squares method. The class searches the directory for 
 * a suitable video file, frame map and tracking data. 
 */
class NIFTKOPENCV_EXPORT HandeyeCalibrateFromDirectory : public mitk::HandeyeCalibrate
{

public:

  mitkClassMacro(HandeyeCalibrateFromDirectory, itk::Object);
  itkNewMacro(HandeyeCalibrateFromDirectory);
  
  void InitialiseVideo ();
  void InitialiseTracking();
     
  itkSetMacro(FramesToUse, unsigned int);
  itkSetMacro(Directory, std::string);
  itkSetMacro(TrackerIndex,int);
  itkSetMacro(AbsTrackerTimingError,long long);

  itkGetMacro(VideoInitialised, bool);
  itkGetMacro(TrackingDataInitialised, bool);
  
  itkSetMacro(NumberCornersWidth, unsigned int);
  itkSetMacro(NumberCornersHeight, unsigned int);
  itkSetMacro(SquareSizeInMillimetres, double);
  itkSetMacro(PixelScaleFactor, mitk::Point2D);
  itkSetMacro(WriteOutChessboards,bool);
  itkSetMacro(NoVideoSupport,bool);
  itkSetMacro(SwapVideoChannels, bool);

protected:

  HandeyeCalibrateFromDirectory();
  virtual ~HandeyeCalibrateFromDirectory();

  HandeyeCalibrateFromDirectory(const HandeyeCalibrateFromDirectory&); // Purposefully not implemented.
  HandeyeCalibrateFromDirectory& operator=(const HandeyeCalibrateFromDirectory&); // Purposefully not implemented.

private:
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
  mitk::Point2D                       m_PixelScaleFactor;
  std::string                         m_Directory;
  bool                                m_WriteOutChessboards;
  bool                                m_WriteOutCalibrationImages;
  mitk::VideoTrackerMatching::Pointer m_Matcher;

  // Not possible to store the frames in memory, will need to process them on the fly
  // First init videotrackermatching
  // use this to get the number of frames (stored in the framemap.log
  // Process frames, either sequencially of more likely by random selection
  std::vector <cv::Mat>               m_LeftCameraVideoFrames;
  std::vector <cv::Mat>               m_RightCameraVideoFrames;

  void                                LoadVideoData(std::string filename);

  /**
   * \brief As video processing can be time consuming allow for saving and loading 
   * of pre-processed video data
   */
  std::string                         CheckForExistingData();

  bool                                m_NoVideoSupport; //for testing, enable running 

  bool                                m_SwapVideoChannels;


}; // end class

} // end namespace

#endif
