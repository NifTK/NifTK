/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkProjectPointsOnStereoVideo_h
#define mitkProjectPointsOnStereoVideo_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <cv.h>
#include <highgui.h>
#include <mitkVideoTrackerMatching.h>

namespace mitk {

/**
 * \class Project points on stereo video
 * \brief Takes an input video file and tracking data. The 
 * video is split into right and left channels.
 * the user can specify a set of points in world coordinates that
 * are projected onto the screen for each frame of the video. 
 * Methods are provided to save:
 * -> the left and right video channels. 
 * -> the projected points as on screen coordinates for each frame
 * -> the points transformed to the coordinate system of the left hand lens.
 *
 */
class NIFTKOPENCV_EXPORT ProjectPointsOnStereoVideo : public itk::Object
{

public:

  mitkClassMacro(ProjectPointsOnStereoVideo, itk::Object);
  itkNewMacro(ProjectPointsOnStereoVideo);

  /** 
   * \brief
   * Set up the projector, finds the video file in the directory, and the tracking data, 
   * and sets up the videotracker matcher
   */
  void Initialise (std::string directory, std::string calibrationParameterDirectory);
  /**
   * \brief
   * performs the point projection
   */
  void  Project();

  /**
   * \brief save the projected coordinates for each frame to a text file
   */
  void SaveProjectedCoordinates (std::string filename);

  /**
   * \brief save the points in left lens coordinates to a text file 
   */
  void SavePointsInLeftLensCoordinates (std::string filename);

  /**
   * \brief Set the world points directly
   */
  void SetWorldPoints (std::vector<cv::Point3f> worldPoints);

  /**
   * \brief Set the world points by triangulating their position from the
   * on screen coordinates for the specified frame
   */
  void SetWorldPointsByTriangulation 
    (std::vector< std::pair<cv::Point2f,cv::Point2f> > onScreenPointPairs, 
     unsigned int FrameNumber);

  void SetVisualise( bool) ;
  void SetSaveVideo( bool);
  itkSetMacro ( TrackerIndex, int);
  itkSetMacro ( DrawLines, bool);
  std::vector < std::vector <cv::Point3f> > GetPointsInLeftLensCS();
  itkGetMacro ( InitOK, bool);
  itkGetMacro ( ProjectOK, bool);

  /**
   * \brief Set the matrix flip state for the VideoTracker matcher
   */
  void SetFlipMatrices (bool);

protected:

  ProjectPointsOnStereoVideo();
  virtual ~ProjectPointsOnStereoVideo();

  ProjectPointsOnStereoVideo(const ProjectPointsOnStereoVideo&); // Purposefully not implemented.
  ProjectPointsOnStereoVideo& operator=(const ProjectPointsOnStereoVideo&); // Purposefully not implemented.

private:
  bool                          m_Visualise; //if true the project function attempts to open a couple of windows to show projection in real time
  bool                          m_SaveVideo; //if true the project function will buffer frames into a object to write out.
  std::string                   m_VideoIn; //the video in file
  std::string                   m_VideoOut; //video needs to be saved on the fly
  std::string                   m_Directory; //the directory containing the data
  std::vector<cv::Point3f>      m_WorldPoints;  //the world points to project

  int                           m_TrackerIndex; //the tracker index to use for frame matching
  mitk::VideoTrackerMatching::Pointer
                                m_TrackerMatcher; //the tracker matcher
 
  bool                          m_DrawLines; //draw lines between the points
  bool                          m_InitOK;
  bool                          m_ProjectOK;

  //the camera calibration parameters
  cv::Mat* m_LeftIntrinsicMatrix;
  cv::Mat* m_LeftDistortionVector;
  cv::Mat* m_RightIntrinsicMatrix;
  cv::Mat* m_RightDistortionVector;
  cv::Mat* m_RightToLeftRotationMatrix;
  cv::Mat* m_RightToLeftTranslationVector;
  cv::Mat* m_LeftCameraToTracker;

  std::vector < std::vector < std::pair<cv::Point2f, cv::Point2f> > >
                                m_ProjectedPoints; // the projected points
  std::vector < std::vector <cv::Point3f> >    
                                m_PointsInLeftLensCS; // the points in left lens coordinates.

  std::vector<std::string> FindVideoData();

  CvCapture*                    m_Capture;
  CvVideoWriter*                m_Writer;
}; // end class

} // end namespace

#endif
