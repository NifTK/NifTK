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
#include <mitkPoint.h>
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

  mitkClassMacroItkParent(HandeyeCalibrateFromDirectory, itk::Object);
  itkNewMacro(HandeyeCalibrateFromDirectory);
  
  void InitialiseOutputDirectory();
  void InitialiseVideo ();
  void InitialiseTracking();
     
  itkSetMacro(FramesToUse, unsigned int);
  itkSetMacro(FramesToCheck, unsigned int);
  itkSetMacro(TrackerIndex,int);
  itkSetMacro(AbsTrackerTimingError,long long);

  itkGetMacro(VideoInitialised, bool);
  itkGetMacro(TrackingDataInitialised, bool);
  
  itkSetMacro(PixelScaleFactor, mitk::Point2D);
  itkSetMacro(WriteOutChessboards,bool);
  itkSetMacro(WriteOutCalibrationImages,bool);
  itkSetMacro(NoVideoSupport,bool);
  itkSetMacro(SwapVideoChannels, bool);
  itkSetMacro(Randomise, bool);

  bool LoadExistingIntrinsicCalibrations (std::string directory);
  bool LoadExistingRightToLeft(const std::string& directory);
  void SetInputDirectory(const std::string& inputDir);
  virtual void SetOutputDirectory(const std::string& outputDir);
  
protected:

  HandeyeCalibrateFromDirectory();
  virtual ~HandeyeCalibrateFromDirectory();

  HandeyeCalibrateFromDirectory(const HandeyeCalibrateFromDirectory&); // Purposefully not implemented.
  HandeyeCalibrateFromDirectory& operator=(const HandeyeCalibrateFromDirectory&); // Purposefully not implemented.

private:
  unsigned int                        m_FramesToUse; //target frames to use actual number used will depend on number of good frames
  unsigned int                        m_FramesToCheck;
  bool                                m_SaveProcessedVideoData;

  bool                                m_VideoInitialised;
  bool                                m_TrackingDataInitialised;

  int                                 m_TrackerIndex;
  long long                           m_AbsTrackerTimingError;
  
  mitk::Point2D                       m_PixelScaleFactor;
  std::string                         m_InputDirectory;
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

  CvMat*                              m_IntrinsicMatrixLeft;
  CvMat*                              m_IntrinsicMatrixRight;
  CvMat*                              m_DistortionCoefficientsLeft;
  CvMat*                              m_DistortionCoefficientsRight;
  CvMat*                              m_RotationMatrixRightToLeft;
  CvMat*                              m_RotationVectorRightToLeft;
  CvMat*                              m_TranslationVectorRightToLeft;
  bool                                m_OptimiseIntrinsics;
  bool                                m_OptimiseRightToLeft;
  bool                                m_Randomise;
}; // end class

} // end namespace

#endif
