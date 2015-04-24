/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkOpenCVPointTypes_h
#define mitkOpenCVPointTypes_h

#include "niftkOpenCVExports.h"
#include <opencv2/opencv.hpp> 
#include <cv.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkTimeStampsContainer.h>

/**
 * \file mitkOpenCVPointTypes.h
 * \brief Derived point types to contain data for projection and analysis
 */
namespace mitk {

/**
* \class contains the gold standard points
* consisting of the frame number, the point and optionally the point index
*/
class NIFTKOPENCV_EXPORT GoldStandardPoint
{
public:

  GoldStandardPoint();
  GoldStandardPoint(unsigned int , int, cv::Point2d);
  GoldStandardPoint(std::istream& is);

  /**
  * \brief an input operator
  */
  friend std::istream& operator>> (std::istream& is, const GoldStandardPoint& gsp );

  friend bool operator < ( const GoldStandardPoint &GSP1 , const GoldStandardPoint &GSP2);

  unsigned int m_FrameNumber;
  int m_Index;
  cv::Point2d  m_Point;
};


/**
 * \class contains a point in 3D and a corresponding scalar value
 */
class NIFTKOPENCV_EXPORT WorldPoint
{
public:

  WorldPoint();
  WorldPoint(cv::Point3d, cv::Scalar);
  WorldPoint(cv::Point3d);

  bool IsNaN ();

  cv::Point3d m_Point;
  cv::Scalar m_Scalar;
};


/**
 * \class contains a vector of world points and a corresponding long long timing error
 */
class NIFTKOPENCV_EXPORT WorldPointsWithTimingError
{
public:

  WorldPointsWithTimingError();
  WorldPointsWithTimingError(std::vector <mitk::WorldPoint>, long long);
  WorldPointsWithTimingError(std::vector <mitk::WorldPoint>);

  std::vector <mitk::WorldPoint> m_Points;
  long long                      m_TimingError;
};


/**
 * \class contains a left and right projected point
 */
class NIFTKOPENCV_EXPORT ProjectedPointPair
{
public:
  ProjectedPointPair();
  ProjectedPointPair(cv::Point2d, cv::Point2d);

  bool LeftNaNOrInf ();
  bool RightNaNOrInf ();
  void SetTimeStamp(const TimeStampsContainer::TimeStamp& ts) { m_TimeStamp = ts; }

  cv::Point2d m_Left;
  cv::Point2d m_Right;
  TimeStampsContainer::TimeStamp m_TimeStamp;
};


/**
 * \class contains a vector of left and right projected points and a timing error
 */
class NIFTKOPENCV_EXPORT ProjectedPointPairsWithTimingError
{
public:
  ProjectedPointPairsWithTimingError();
  ProjectedPointPairsWithTimingError(std::vector <mitk::ProjectedPointPair>, long long);
  ProjectedPointPairsWithTimingError(std::vector <mitk::ProjectedPointPair>);

  std::vector <mitk::ProjectedPointPair> m_Points;
  long long                              m_TimingError;
};


/**
 * \class contains an opencv matrix of video data, a left or right flag and the
 * timestamp
 */
class NIFTKOPENCV_EXPORT VideoFrame
{
public:

  VideoFrame();
  VideoFrame(cv::VideoCapture* capture, std::ifstream* frameMapLogFile);

  itkGetMacro   (VideoData, cv::Mat);
  itkGetMacro   (TimeStamp, unsigned long long);

  bool WriteToFile (std::string prefix);
  void OutputVideoInformation (cv::VideoCapture* capture);

private:

  cv::Mat             m_VideoData;
  unsigned long long  m_TimeStamp;
  bool                m_Left;

  unsigned int        m_FrameNumber;
  unsigned int        m_SequenceNumber;
  unsigned int        m_Channel;
};

/**
 * \class contains a vector of 2D points.
 */
class NIFTKOPENCV_EXPORT PickedObject
{
  public:
      
    int id;
    bool isLine;
    std::vector < cv::Point2d > points;

    PickedObject();
    ~PickedObject();
};

/**
 * \class maintains a set a point vectors and ID's that 
 * can be used to represent lines or points in an image
 */
class NIFTKOPENCV_EXPORT PickedPointList : public itk::Object
{
public:
  mitkClassMacro(PickedPointList, itk::Object);
  itkNewMacro(PickedPointList);

  void PutOut (std::ofstream& os);
  void AnnotateImage (cv::Mat& image);
  cv::Mat CreateMaskImage ( const cv::Mat& image);

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
* \brief a call back function for dealing with PickedPointLists
*/
void PointPickingCallBackFunc (  int, int , int, int, void* );

} // end namespace

#endif



