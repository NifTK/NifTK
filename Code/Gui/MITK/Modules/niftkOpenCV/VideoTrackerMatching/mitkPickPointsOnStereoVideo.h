/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkPickPointsOnStereoVideo_h
#define mitkPickPointsOnStereoVideo_h

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
 * \brief the call back function for mouse events events during point picking
 */
void PointPickingCallBackFunc (  int, int , int, int, void* );

class PickedObject 
{
public:
  int id;
  bool isLine;
  std::vector < cv::Point2d > points;

  PickedObject();
  ~PickedObject();
};

class PickedPointList : public itk::Object
{
  public:
    mitkClassMacro(PickedPointList, itk::Object);
    itkNewMacro(PickedPointList);
    
    void PutOut (std::ofstream& os);
    void AnnotateImage (cv::Mat& image);

    void SetInLineMode (const bool& mode);
    void SetInOrderedMode ( const bool& mode);
    bool GetIsModified();
    itkSetMacro (FrameNumber, unsigned int);
    itkSetMacro (Channel, std::string);

    unsigned int AddPoint (const cv::Point2d& point);
    unsigned int RemoveLastPoint ();
    unsigned int SkipOrderedPoint ();
    unsigned int EndLine();

  protected:
    PickedPointList();
    virtual ~PickedPointList();

    PickedPointList (const PickedPointList&); // Purposefully not implemented.
    PickedPointList& operator=(const PickedPointList&); // Purposefully not implemented.

  private:
    bool m_InLineMode;
    bool m_InOrderedMode;
    bool m_IsModified;
    unsigned int m_FrameNumber;
    std::string m_Channel;
    std::vector < PickedObject > m_PickedObjects;
    int GetNextAvailableID ( bool ForLine );
};
/**
 * \class Pick points in stereo video
 * \brief Takes an input video (.264) file and tracking data. The 
 * video is split into right and left channels.
 * the user can specifies how many points to pick and frequency of 
 * point picked frames. The class reads through the 
 * video file and at set intervals provides and interface to select 
 * a set number of points in the frame. Frames without matching 
 * tracking data are not processed.
 * The picked points are out put as the onscreen coordinates for each 
 * frame, and if matching points are present they are triangulated.
 */
class NIFTKOPENCV_EXPORT PickPointsOnStereoVideo : public itk::Object
{

public:

  mitkClassMacro(PickPointsOnStereoVideo, itk::Object);
  itkNewMacro(PickPointsOnStereoVideo);

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
  void  Project(mitk::VideoTrackerMatching::Pointer matcher);
  
  /**
   * \brief
   * Sets the cameratotracker matrix for the passed matcher 
   * to match the matrix for the projector
   */
  void  SetMatcherCameraToTracker(mitk::VideoTrackerMatching::Pointer matcher);

  itkSetMacro ( TrackerIndex, int);
  itkSetMacro ( ReferenceIndex, int);
  itkSetMacro ( AllowableTimingError, long long);
  itkSetMacro ( OrderedPoints, bool);
  itkSetMacro ( PickingLine, bool);
  itkSetMacro ( AskOverWrite, bool);
  itkSetMacro ( HaltOnVideoReadFail, bool);
  itkSetMacro ( WriteAnnotatedImages, bool);
  itkSetMacro ( Frequency, unsigned int);

  itkGetMacro ( InitOK, bool);
  itkGetMacro ( ProjectOK, bool);
  itkGetMacro ( WorldToLeftCameraMatrices, std::vector < cv::Mat > );
 
protected:

  PickPointsOnStereoVideo();
  virtual ~PickPointsOnStereoVideo();

  PickPointsOnStereoVideo(const PickPointsOnStereoVideo&); // Purposefully not implemented.
  PickPointsOnStereoVideo& operator=(const PickPointsOnStereoVideo&); // Purposefully not implemented.

private:
  std::string                   m_VideoIn; //the video in file
  std::string                   m_Directory; //the directory containing the data

  int                           m_TrackerIndex; //the tracker index to use for frame matching
  int                           m_ReferenceIndex; //the reference index to use for frame matching, not used by default
 
  bool                          m_InitOK;
  bool                          m_ProjectOK;
  bool                          m_OrderedPoints; //picked points can be ordered or unordered
  bool                          m_PickingLine; //if true we are picking a line defined by a vector of points
  bool                          m_AskOverWrite; //if true, we will ask if you want to overwrite existing results
  bool                          m_HaltOnVideoReadFail; //halt if video read fail
  bool                          m_WriteAnnotatedImages; //halt if video read fail

  unsigned int                  m_StartFrame; //you can exclude some frames at the start
  unsigned int                  m_EndFrame; // and at the end
  unsigned int                  m_Frequency; // the sample rate (process every m_Frequency frame)


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

  std::vector < cv::Mat >       m_WorldToLeftCameraMatrices;    // the saved camera positions

  cv::VideoCapture*             m_Capture;

  
  long long                     m_AllowableTimingError; // the maximum permisable timing error when setting points or calculating projection errors;
  
}; // end class

} // end namespace

#endif
