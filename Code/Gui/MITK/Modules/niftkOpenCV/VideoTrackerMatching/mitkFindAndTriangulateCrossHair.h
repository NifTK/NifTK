/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkFindAndTriangulateCrossHair_h
#define mitkFindAndTriangulateCrossHair_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <cv.h>
#include <highgui.h>
#include "mitkVideoTrackerMatching.h"

namespace mitk {

/**
 * \class 
 * \brief
 * Takes an input video of a cross hair and
 * determines the centre point of the cross hair in 
 * screen coordinates. Given the camera calibration 
 * parameters and optionally the tracking data the 
 * cross hair centre can be triangulated to left lens or 
 * world coordinates
 *
 */
class NIFTKOPENCV_EXPORT FindAndTriangulateCrossHair : public itk::Object
{

public:

  mitkClassMacro(FindAndTriangulateCrossHair, itk::Object);
  itkNewMacro(FindAndTriangulateCrossHair);

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
  void Triangulate();

  void SetVisualise( bool) ;
  void SetSaveVideo( bool);
  itkSetMacro ( TrackerIndex, int);
  itkSetMacro ( FramesToProcess, int);
  itkSetMacro ( HaltOnVideoReadFail, bool);
  itkGetMacro ( PointsInLeftLensCS, std::vector<mitk::WorldPoint> );
  itkGetMacro ( WorldPoints, std::vector<mitk::WorldPoint> );
  itkGetMacro ( ScreenPoints, std::vector<mitk::ProjectedPointPair> );
  itkGetMacro ( InitOK, bool);
  itkGetMacro ( TriangulateOK, bool);

  /**
   * \brief Set the matrix flip state for the VideoTracker matcher
   */
  void SetFlipMatrices (bool);

  /** 
  * \brief set the video lag parameters for the tracker matcher
  */
  void SetVideoLagMilliseconds (unsigned long long videoLag, bool videoLeadsTracking = false);

protected:

  FindAndTriangulateCrossHair();
  virtual ~FindAndTriangulateCrossHair();

  FindAndTriangulateCrossHair(const FindAndTriangulateCrossHair&); // Purposefully not implemented.
  FindAndTriangulateCrossHair& operator=(const FindAndTriangulateCrossHair&); // Purposefully not implemented.

private:
  bool                          m_Visualise; //if true the project function attempts to open a couple of windows to show projection in real time
  bool                          m_SaveVideo; //if true the project function will buffer frames into a object to write out.
  std::string                   m_VideoIn; //the video in file
  std::string                   m_VideoOut; //video needs to be saved on the fly
  std::string                   m_Directory; //the directory containing the data

  std::vector<mitk::WorldPoint> m_WorldPoints;  //the triangulated points in world coordinates
  std::vector<mitk::WorldPoint> m_PointsInLeftLensCS;  //the triangulated points in world coordinates
  int                           m_TrackerIndex; //the tracker index to use for frame matching
  mitk::VideoTrackerMatching::Pointer
                                m_TrackerMatcher; //the tracker matcher
 
  bool                          m_InitOK;
  bool                          m_TriangulateOK;

  int                           m_FramesToProcess; //can stop early, if negative it runs to the end

  //the camera calibration parameters
  cv::Mat* m_LeftIntrinsicMatrix;
  cv::Mat* m_LeftDistortionVector;
  cv::Mat* m_RightIntrinsicMatrix;
  cv::Mat* m_RightDistortionVector;
  cv::Mat* m_RightToLeftRotationMatrix;
  cv::Mat* m_RightToLeftTranslationVector;
  cv::Mat* m_LeftCameraToTracker;
  //the video screen dimensions
  double   m_VideoWidth;
  double   m_VideoHeight;
  double   m_DefaultVideoWidth;
  double   m_DefaultVideoHeight;

  std::vector < mitk::ProjectedPointPair > 
                                m_ScreenPoints; // the projected points

  cv::VideoCapture*             m_Capture;
  bool                          m_HaltOnVideoReadFail;
  CvVideoWriter*                m_Writer;

  cv::Size                      m_BlurKernel; //for blurring
 
  double                        m_HoughRho; //for the hough filter
  double                        m_HoughTheta; //for the hough filter
  int                           m_HoughThreshold; //for the hough filter
  int                           m_HoughLineLength; //for the hough filter
  int                           m_HoughLineGap; //for the hough filter
  void TriangulatePoints();
  void TransformPointsToWorld();

}; // end class

} // end namespace

#endif
