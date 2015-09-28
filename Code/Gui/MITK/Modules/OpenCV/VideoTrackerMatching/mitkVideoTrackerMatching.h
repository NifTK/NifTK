/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkVideoTrackerMatching_h
#define mitkVideoTrackerMatching_h

#include "niftkOpenCVExports.h"
#include <mitkOpenCVPointTypes.h>
#include <cv.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include "mitkTrackingMatrices.h"
#include "mitkTimeStampsContainer.h"

namespace mitk
{

/**
 * \brief A class to match video frames to tracking frames, when reading recorded tracking data. 
 */
class NIFTKOPENCV_EXPORT VideoTrackerMatching : public itk::Object
{
public: 
  mitkClassMacroItkParent ( VideoTrackerMatching, itk::Object);
  itkNewMacro (VideoTrackerMatching);

  /**
   * \brief Initialise the class by passing it a directory name.
   */
  void Initialise (std::string directory);

  /**
   * \brief Return the tracking matrix for a given video frame number.
   */
  cv::Mat GetTrackerMatrix ( unsigned int FrameNumber, long long * TimingError = NULL, unsigned int TrackerIndex = 0 );

  /**
   * \brief Return the tracking matrix multiplied by the camera to tracker matrix for a given video frame number.
   */
  cv::Mat GetCameraTrackingMatrix ( unsigned int FrameNumber, long long * TimingError = NULL, unsigned int TrackerIndex = 0 , std::vector <double> * Perturbation = NULL , int ReferenceIndex = -1 );

  /**
   * \brief Returns a frame of video data (WARNING, not implemented, if you use this you will only get a 4x4 junk matrix back) and the time stamp for a given frame number
   */
  cv::Mat GetVideoFrame ( unsigned int FrameNumber, unsigned long long * TimingStamp = NULL );

  /**
   * \brief returns state of m_Ready
   */
  bool IsReady () 
  { 
    return m_Ready;
  } 
  
  /**
   * \brief Set a flag as to whether to flip matrices between left/right handed, default is right.
   */
  itkSetMacro (FlipMatrices, bool);

  /**
   * \brief Set a flag as to whether to write out the timing errors
   */
  itkSetMacro (WriteTimingErrors, bool);

  /**
   * \brief Set a flag to determine what to do if a skipped frame is found, by default we halt
   */
  itkSetMacro (HaltOnFrameSkip, bool);
  
  /**
   * \brief Set a flag to determine what to do if a skipped frame is found, by default we halt
   */
  itkGetMacro (FrameMap, std::string);

  /**
   * \brief Get the frame count.
   */
  int GetNumberOfFrames () 
  {
    return m_FrameNumbers.size();
  }

  /**
   * \brief If the tracking data is ahead of the video data you can set a video lag in 
   * milliseconds to account for this. If the video is ahead of the tracking set 
   * argument 2 to true
   */
  void SetVideoLagMilliseconds(unsigned long long VideoLag, bool VideoLeadsTracking =false, int trackerIndex = -1);

  /**
   * \brief set the camera to tracker matrix. if tracker index = -1 all camera to tracker 
   * matrices will be set with the same value
   */
  void SetCameraToTracker( cv::Mat, int trackerIndex = -1 );

  /*
   * \brief Convienient way to set the camera to tracker matrices where multiple tracking 
   * directories are present. File contains camera to tracker matrices in same order as 
   * tracker indices
   */
  void SetCameraToTrackers ( std::string filename );

  /*
   * \brief Return the size of the TrackingMatrices vector (how many tracker indeces are available)
   */
  unsigned int GetTrackingMatricesSize ();


protected:
  VideoTrackerMatching();
  virtual ~VideoTrackerMatching();

  VideoTrackerMatching(const VideoTrackerMatching&); // Purposefully not implemented.
  VideoTrackerMatching& operator=(const VideoTrackerMatching&); // Purposefully not implemented.
  
  std::vector<unsigned int>             m_FrameNumbers;
  std::vector<TimeStampsContainer> m_TimeStampsContainer;
  TimeStampsContainer              m_VideoTimeStamps;
  bool                                  m_Ready;
  bool                                  m_FlipMatrices;
  bool                                  m_WriteTimingErrors;
  std::string                           m_Directory;

  /**
   * \brief Reads a file that defines the position of a point fixed in world
   * coordinates relative to the camera lens.
   * [framenumber][pointID]
   * [framenumber][pointID](left,right)
   */
  std::vector < mitk::WorldPointsWithTimingError >  ReadPointsInLensCSFile (std::string filename, 
      int PointsPerFrame = 1 ,
      std::vector < mitk::ProjectedPointPairsWithTimingError >* onScreenPoints = NULL);
  
private:
  
  std::vector<TrackingMatrices>     m_TrackingMatrices; 
  std::vector<std::string>          m_TrackingMatrixDirectories;
  std::string                       m_FrameMap;

  std::vector<std::string>          FindFrameMaps();
  void                              FindTrackingMatrixDirectories();
  void                              ProcessFrameMapFile();
  bool                              CheckTimingErrorStats();
  bool                              m_HaltOnFrameSkip;
  std::vector<cv::Mat>              m_CameraToTracker;

  std::vector <unsigned long long> m_VideoLag; //the delay between the tracking and video data
  std::vector <bool>               m_VideoLeadsTracking; //if the video lag is negative, set this to true

};


} // namespace


#endif // niftkVideoTrackerMatching_h
