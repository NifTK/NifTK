/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkVideoToSurface_h
#define mitkVideoToSurface_h

#include "niftkOpenCVExports.h"
#include <mitkOpenCVPointTypes.h>
#include <string>
#include <fstream>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <cv.h>
#include <highgui.h>
#include "mitkVideoTrackerMatching.h"

namespace mitk {

/**
 * \class Video to surface
 * \brief Takes an input video file and tracking data. The 
 * video is split into right and left channels.
 * For each frame pair, features a matched and
 * triangulated.
 *
 */
class NIFTKOPENCV_EXPORT VideoToSurface : public itk::Object
{

public:

  mitkClassMacroItkParent(VideoToSurface, itk::Object);
  itkNewMacro(VideoToSurface);

  /** 
   * \brief
   * Set up the projector, finds the video file in the directory, and the tracking data, 
   * and sets up the videotracker matcher
   */
  void Initialise (std::string directory, std::string calibrationParameterDirectory);
 /** 
   * \brief
   * Set up the projector, finds the video file in the directory, and the tracking data, 
   * and sets up the videotracker matcher, without any calibration information
   */
  void Initialise (std::string directory);
  
  /**
   * \brief
   * performs the point projection
   */
  void  Reconstruct(mitk::VideoTrackerMatching::Pointer matcher);
  
  /**
   * \brief
   * Sets the cameratotracker matrix for the passed matcher 
   * to match the matrix for the projector
   */
  void  SetMatcherCameraToTracker(mitk::VideoTrackerMatching::Pointer matcher);

  void SetSaveVideo( bool state );
  itkSetMacro ( TrackerIndex, int);
  itkSetMacro ( ReferenceIndex, int);
  itkSetMacro ( EndFrame, unsigned int);
  itkSetMacro ( HaltOnVideoReadFail, bool);
  itkSetMacro ( VisualiseTrackingStatus, bool);
  itkSetMacro ( AllowableTimingError, long long);

  itkGetMacro ( InitOK, bool);
  itkGetMacro ( WorldToLeftCameraMatrices, std::vector < cv::Mat > );

protected:

  VideoToSurface();
  virtual ~VideoToSurface();

  VideoToSurface(const VideoToSurface&); // Purposefully not implemented.
  VideoToSurface& operator=(const VideoToSurface&); // Purposefully not implemented.

private:
  bool                          m_SaveVideo; //if true the project function will buffer frames into a object to write out.
  std::string                   m_VideoIn; //the video in file
  std::string                   m_VideoOut; //video needs to be saved on the fly
  std::string                   m_Directory; //the directory containing the data
  std::string                   m_OutDirectory; //where to write out any video

  int                           m_TrackerIndex; //the tracker index to use for frame matching
  int                           m_ReferenceIndex; //the reference index to use for frame matching, not used by default
 
  bool                          m_InitOK;
  bool                          m_HaltOnVideoReadFail; //stop processing if video read fails
  bool                          m_VisualiseTrackingStatus; //draw something on screen to indicate whether tracking was working got frame

  unsigned int                  m_StartFrame; //you can exclude some frames at the start
  unsigned int                  m_EndFrame; // and at the end

  double                        m_TriangulationTolerance;
  
  //the camera calibration parameters
  cv::Mat* m_LeftIntrinsicMatrix;
  cv::Mat* m_LeftDistortionVector;
  cv::Mat* m_RightIntrinsicMatrix;
  cv::Mat* m_RightDistortionVector;
  cv::Mat* m_RightToLeftRotationMatrix;
  cv::Mat* m_RightToLeftTranslationVector;
  cv::Mat* m_LeftCameraToTracker;

  //the dimensions of the video screen in pixels
  double   m_VideoWidth;
  double   m_VideoHeight;

  //the dimensions of the patch to reconstruct in pixels, and its offset from origin
  double   m_PatchWidth;
  double   m_PatchHeight;
  double   m_PatchOriginX;
  double   m_PatchOriginY;

  //defaults for generating a reconstruction depth histogram
  double   m_HistogramMaximumDepth;
  
  std::vector < cv::Mat >       m_WorldToLeftCameraMatrices;    // the saved camera positions

  cv::VideoCapture*             m_Capture;
  CvVideoWriter*                m_LeftWriter;
  CvVideoWriter*                m_RightWriter;

  long long                     m_AllowableTimingError; // the maximum permisable timing error when setting points or calculating projection errors;

  void FindVideoData (mitk::VideoTrackerMatching::Pointer trackerMatcher);

  //returns a patch of image, based on patch definition above
  cv::Mat GetPatch ( const cv::Mat& image );

  //annotates the image with patch, recconstruction, and tracking statistics
  
  void AnnotateImage ( cv::Mat& image, const cv::Mat& patch, 
      const long long& timingError,
      const double& patchDepthMean,
      const double& patchDepthStdDev,
      const unsigned int& patchVectorSize,
      const std::vector < unsigned int >& patchDepthHistogram,
      const double& meanTriangulationError );
  
}; // end class

} // end namespace

#endif
