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

#include "niftkOpenCVUtilsExports.h"
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

// forward declaration
class GoldStandardPoint;

extern "C++" NIFTKOPENCVUTILS_EXPORT
std::istream& operator>> (std::istream& is, const GoldStandardPoint& gsp );

extern "C++" NIFTKOPENCVUTILS_EXPORT
bool operator < ( const GoldStandardPoint &GSP1 , const GoldStandardPoint &GSP2);

/**
* \brief a call back function for dealing with PickedPointLists
*/
extern "C++" NIFTKOPENCVUTILS_EXPORT
void PointPickingCallBackFunc (  int, int , int, int, void* );


/**
* \class contains the gold standard points
* consisting of the frame number, the point and optionally the point index
*/
class NIFTKOPENCVUTILS_EXPORT GoldStandardPoint
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
class NIFTKOPENCVUTILS_EXPORT WorldPoint
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
class NIFTKOPENCVUTILS_EXPORT WorldPointsWithTimingError
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
class NIFTKOPENCVUTILS_EXPORT ProjectedPointPair
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
class NIFTKOPENCVUTILS_EXPORT ProjectedPointPairsWithTimingError
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
class NIFTKOPENCVUTILS_EXPORT VideoFrame
{
public:

  VideoFrame();
  VideoFrame(cv::VideoCapture* capture, std::ifstream* frameMapLogFile);

  itkGetMacro   (VideoData, cv::Mat);
  itkGetMacro   (TimeStamp, unsigned long long);

  bool WriteToFile (const std::string& prefix, const std::string& fileExtension="bmp");
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
 * \class contains a vector of 3D points, an identifier and whether or not it's a line
 */
class NIFTKOPENCVUTILS_EXPORT PickedObject
{
  public:

    int m_Id;
    bool m_IsLine;
    std::vector < cv::Point3d > m_Points;
    unsigned int m_FrameNumber;
    unsigned long long m_TimeStamp;
    std::string m_Channel;
    cv::Scalar m_Scalar;

    PickedObject();
    PickedObject(std::string channel, unsigned int framenumber, unsigned long long timestamp, cv::Scalar scalar);
    PickedObject(const GoldStandardPoint& gsp, const unsigned long long& timestamp); //cast a gold standard point to a PickedObject
    ~PickedObject();

    /**
     * \brief compare the header information (Id, IsLine, Channel, FrameNumber)
     * and return true if they all match, except if m_Id in otherPickedObject is -1, which acts
     * as a wildcard
     */
    bool HeadersMatch ( const PickedObject& otherPickedObject, const long long& allowableTimingError = 20e6) const;

    /**
     * \brief Calculates a distance between two picked objects
     * returns infinity if the headers don't match.
     * Delta's contains the
     * signed distance with a header matching the calling object
     * The first value in m_Points is the signed distance,
     * the second value is the centroid of the calling objects m_Points,
     * the third value  is the centroid of the otherPickedObject
     */
    double DistanceTo ( const PickedObject& otherPickedObject, PickedObject& deltas, const long long& allowableTimingError = 20e6) const;

    /**
     * \brief Copy the header information to a new instance
     */
    PickedObject CopyByHeader () const;

};

std::istream& operator >> ( std::istream& is, PickedObject& po);

extern "C++" NIFTKOPENCVUTILS_EXPORT bool operator < ( const PickedObject &PO1 , const PickedObject &PO2);
extern "C++" NIFTKOPENCVUTILS_EXPORT PickedObject operator * ( const PickedObject &PO1 , const cv::Mat* transform);

/**
 * \class maintains a set a point vectors and ID's that
 * can be used to represent lines or points in an image
 */
class NIFTKOPENCVUTILS_EXPORT PickedPointList : public itk::Object
{
public:
  mitkClassMacroItkParent(PickedPointList, itk::Object)
  itkNewMacro(PickedPointList)

  void PutOut (std::ofstream& os);
  void AnnotateImage (cv::Mat& image, int lineThickness);
  cv::Mat CreateMaskImage ( const cv::Mat& image);

  void SetInLineMode (const bool& mode);
  void SetInOrderedMode ( const bool& mode);
  bool GetIsModified();
  itkSetMacro (FrameNumber, unsigned int);
  itkGetConstMacro (FrameNumber, unsigned int);
  itkSetMacro (Channel, std::string);
  itkGetMacro (Channel, std::string);
  itkSetMacro (TimeStamp, unsigned long long);
  itkGetMacro (TimeStamp, unsigned long long);
  itkSetMacro (XScale, double);
  itkSetMacro (YScale, double);
  std::vector <mitk::PickedObject> GetPickedObjects() const;
  void SetPickedObjects ( const std::vector < mitk::PickedObject > & objects );

  unsigned int GetListSize() const;
  unsigned int GetNumberOfPoints() const;
  unsigned int GetNumberOfLines() const;
  void ClearList();

  unsigned int AddPoint (const cv::Point2i& point);
  unsigned int AddPoint (const cv::Point3d& point, cv::Scalar scalar);
  unsigned int RemoveLastPoint ();
  unsigned int SkipOrderedPoint ();

  /**
   * \brief checks the picked object vector to see if it contains a picked object of same type and ID as target
   * if not it adds a dummy point to the vector
   */
  void AddDummyPointIfNotPresent ( const mitk::PickedObject& target );

  mitk::PickedPointList::Pointer CopyByHeader();
  mitk::PickedPointList::Pointer TransformPointList (cv::Mat* transform);

protected:
  PickedPointList();
  virtual ~PickedPointList();

  PickedPointList (const PickedPointList&); // Purposefully not implemented.
  PickedPointList& operator=(const PickedPointList&); // Purposefully not implemented.

private:
  bool m_InLineMode;
  bool m_InOrderedMode;
  bool m_IsModified;
  double m_XScale; // When adding points, we can scale the x pixel location
  double m_YScale; // When adding points, we can scale the y pixel location

  unsigned long long m_TimeStamp;
  unsigned int m_FrameNumber;
  std::string m_Channel;
  std::vector < PickedObject > m_PickedObjects;
  int GetNextAvailableID ( bool ForLine );
};
/**
 * \brief a function to cast a point3d to a point2i, checks that z is zero, throws an error is not
 */
cv::Point2i NIFTKOPENCVUTILS_EXPORT Point3dToPoint2i (const cv::Point3d& point);

} // end namespace

#endif



