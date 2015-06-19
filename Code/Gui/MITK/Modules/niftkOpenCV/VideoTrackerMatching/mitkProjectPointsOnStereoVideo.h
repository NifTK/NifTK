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
   * Set up the projector, finds the video file in the directory, and the tracking data, 
   * and sets up the videotracker matcher, without any calibration information
   */
  void Initialise (std::string directory);
  
  /**
   * \brief
   * performs the point projection
   */
  void  Project(mitk::VideoTrackerMatching::Pointer matcher, std::vector<double> * perturbation = NULL);
  
  /**
   * \brief
   * Sets the cameratotracker matrix for the passed matcher 
   * to match the matrix for the projector
   */
  void  SetMatcherCameraToTracker(mitk::VideoTrackerMatching::Pointer matcher);

  /**
   * \brief save the projected coordinates for each frame to a text file
   */
  void SaveProjectedCoordinates (std::string filename);

  /**
   * \brief save the points in left lens coordinates to a text file 
   */
  void SavePointsInLeftLensCoordinates (std::string filename);

  /**
   * \brief Append to world points by triangulating their position from the
   * on screen coordinates for the specified frame
   */
  void AppendWorldPointsByTriangulation 
    (std::vector< mitk::ProjectedPointPair > onScreenPointPairs, 
     std::vector < unsigned int>  frameNumber , mitk::VideoTrackerMatching::Pointer matcher, 
     std::vector <double> * perturbation = NULL);

  void SetVisualise( bool) ;
  void SetSaveVideo( bool state, std::string prefix = "" );
  itkSetMacro ( TrackerIndex, int);
  itkSetMacro ( ReferenceIndex, int);
  itkSetMacro ( DrawLines, bool);
  itkSetMacro ( DrawAxes, bool);
  itkSetMacro ( HaltOnVideoReadFail, bool);
  itkSetMacro ( DontProject, bool);
  itkSetMacro ( VisualiseTrackingStatus, bool);
  itkSetMacro ( AllowablePointMatchingRatio, double);
  itkSetMacro ( AllowableTimingError, long long);
  void SetLeftGoldStandardPoints ( std::vector <GoldStandardPoint> points );
  void SetRightGoldStandardPoints ( std::vector <GoldStandardPoint > points );

  /**
   * \brief appends points to  world points and corresponding vectors
   */
  void AppendWorldPoints ( std::vector< mitk::WorldPoint > points );
 
  /** 
   * \brief appends point to classifier world points
   */
  void AppendClassifierWorldPoints ( std::vector < mitk::WorldPoint > points );
  /** 
   * \brief clear the list of world points
   */
  void ClearWorldPoints ();

  itkGetMacro ( PointsInLeftLensCS, std::vector <mitk::WorldPointsWithTimingError> );
  itkGetMacro ( ProjectedPoints, std::vector <mitk::ProjectedPointPairsWithTimingError>);
  itkGetMacro ( InitOK, bool);
  itkGetMacro ( ProjectOK, bool);
  itkGetMacro ( WorldToLeftCameraMatrices, std::vector < cv::Mat > );
  itkGetMacro ( LeftProjectionErrors, std::vector < cv::Point2d > ); 
  itkGetMacro ( RightProjectionErrors, std::vector < cv::Point2d > );  
  itkGetMacro ( LeftReProjectionErrors, std::vector < cv::Point3d > ); 
  itkGetMacro ( RightReProjectionErrors, std::vector < cv::Point3d > );  
  itkGetMacro ( TriangulationErrors, std::vector < cv::Point3d > );  
 
  /**
   * \brief calculates the projection and re-projection errors
   */
  void CalculateProjectionErrors (std::string outPrefix);

  /**
   * \brief calculates the triangulation errors
   */
  void CalculateTriangulationErrors (std::string outPrefix,  mitk::VideoTrackerMatching::Pointer trackerMatcher);


  /** 
   * \brief Set the projector screen buffer
   */
  itkSetMacro ( ProjectorScreenBuffer, double);

  itkSetMacro ( ClassifierScreenBuffer, double);
  itkSetMacro ( TriangulatedPointsOutName, std::string );
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
  std::string                   m_VideoOutPrefix; //where to write out any video
  std::string                   m_TriangulatedPointsOutName; //where to write the triangulated points out to
  std::vector< mitk::WorldPoint >     
                                m_WorldPoints;  //the world points to project, and their accompanying scalar values 

  int                           m_TrackerIndex; //the tracker index to use for frame matching
  int                           m_ReferenceIndex; //the reference index to use for frame matching, not used by default
 
  bool                          m_DrawLines; //draw lines between the points
  bool                          m_InitOK;
  bool                          m_ProjectOK;
  bool                          m_DrawAxes;
  bool                          m_LeftGSFramesAreEven; // true if the left GS frame numbers are even
  bool                          m_RightGSFramesAreEven; // true if the right GS frame numbers are even
  bool                          m_HaltOnVideoReadFail; //stop processing if video read fails
  bool                          m_DontProject; //don't project anything, useful for just reviewing video data
  bool                          m_VisualiseTrackingStatus; //draw something on screen to indicate whether tracking was working got frame
  int                           m_RightGSFrameOffset; //0 if right and left gold standard points have the same frame number 
  int                           m_MaxGoldStandardIndex; //useful if we're just triangulating gold standard points

  unsigned int                  m_StartFrame; //you can exclude some frames at the start
  unsigned int                  m_EndFrame; // and at the end

  double                        m_ProjectorScreenBuffer; // A buffer around the screen beyond which projected points will be set to infinity
  double                        m_ClassifierScreenBuffer; // A buffer around the screen, beyond which projected classifier points will be set to infinity

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

  /* m_ProjectPoints [framenumber](timingError,[pointID](left.right));*/
  std::vector < mitk::ProjectedPointPairsWithTimingError >
                                m_ProjectedPoints; // the projected points
  std::vector < mitk::WorldPointsWithTimingError >    
                                m_PointsInLeftLensCS; // the points in left lens coordinates.
  mitk::ProjectedPointPairsWithTimingError 
                                m_ScreenAxesPoints; // the projected axes points

  std::vector < cv::Mat >       m_WorldToLeftCameraMatrices;    // the saved camera positions

  // a bunch of stuff for calculating errors
  std::vector < mitk::GoldStandardPoint >
                                m_LeftGoldStandardPoints;   //for calculating errors, the gold standard left screen points
  std::vector < mitk::GoldStandardPoint >
                                m_RightGoldStandardPoints;   //for calculating errors, the gold standard right screen points
  std::vector<mitk::WorldPoint> m_ClassifierWorldPoints;  //the world points to project, to classify the gold standard screen points
  std::vector < mitk::ProjectedPointPairsWithTimingError >
                                m_ClassifierProjectedPoints; // the projected points used for classifying the gold standard screen points

  std::vector < cv::Point2d >   m_LeftProjectionErrors;  //the projection errors in pixels
  std::vector < cv::Point2d >   m_RightProjectionErrors;  //the projection errors in pixels
  std::vector < cv::Point3d >   m_LeftReProjectionErrors; // the projection errors in mm reprojected onto a plane normal to the camera lens
  std::vector < cv::Point3d >   m_RightReProjectionErrors; // the projection errors in mm reprojected onto a plane normal to the camera lens
  std::vector < cv::Point3d >   m_TriangulationErrors; // the projection errors in mm reprojected onto a plane normal to the camera lens

  cv::VideoCapture*             m_Capture;
  CvVideoWriter*                m_LeftWriter;
  CvVideoWriter*                m_RightWriter;

  double                        m_AllowablePointMatchingRatio; // the minimum allowable ratio between the 2 nearest points when matching points on screen
  
  long long                     m_AllowableTimingError; // the maximum permisable timing error when setting points or calculating projection errors;
  void ProjectAxes();

  /* \brief 
   * calculates the x and y errors between the passed point and the nearest point in 
   * m_ProjectedPoints, adds result to m_LeftProjectionErrors or m_RightProjectionErrors
   */
  void CalculateProjectionError (  GoldStandardPoint GSPoint, bool left );

  /* \brief 
   * calculates the x,y, and z error between the passed point and the nearest point in 
   * m_ProjectedPoints when projected onto a plane distant from the camera
   * appends result to m_LeftReProjectionErrors or m_RightReProjectionErrors
   */
  void CalculateReProjectionError ( GoldStandardPoint GSPoint, bool left );
 
  /* \brief 
   * Finds  the nearest point in 
   * m_ProjectedPoints
   */
  cv::Point2d FindNearestScreenPoint ( GoldStandardPoint GSPoint, 
      bool left,  double* minRatio = NULL ,unsigned int * index = NULL );
  
}; // end class

} // end namespace

#endif
