/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkOpenCVPointTypes.h"
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/special_functions/round.hpp>
#include <numeric>
#include <algorithm>
#include <functional>
#include <mitkExceptionMacro.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>
#include <string>
#include <fstream>

namespace mitk {

//-----------------------------------------------------------------------------
GoldStandardPoint::GoldStandardPoint()
: m_FrameNumber(0)
, m_Index (-1)
, m_Point (cv::Point2d( std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()))
{}


//-----------------------------------------------------------------------------
GoldStandardPoint::GoldStandardPoint(unsigned int framenumber, int index, cv::Point2d point)
: m_FrameNumber(framenumber)
, m_Index (index)
, m_Point (point)
{}


//-----------------------------------------------------------------------------
GoldStandardPoint::GoldStandardPoint( std::istream &is)
: m_FrameNumber(0)
, m_Index (-1)
, m_Point (cv::Point2d( std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()))
{
  std::string line;
  if ( std::getline(is,line) )
  {
    std::stringstream linestream(line);
    bool parseSuccess;
    double parse[4];
    parseSuccess = linestream >> parse[0] >> parse[1] >> parse[2] >> parse[3];
    if ( parseSuccess )
    {
      m_FrameNumber = static_cast<unsigned int> (parse[0]);
      m_Index = static_cast<int>(parse[1]);
      m_Point.x = parse[2];
      m_Point.y = parse[3];
      return;
    }
    else
    {
      std::stringstream linestream2(line);
      parseSuccess = linestream2 >> parse[0] >> parse[1] >> parse[2];
      if ( parseSuccess )
      {
        m_FrameNumber = static_cast<unsigned int> (parse[0]);
        m_Point.x = parse[1];
        m_Point.y = parse[2];
        m_Index = -1;
      }
      else
      {
        MITK_WARN << "Error reading gold standard point";
      }
    }
  }
  else
  {
    MITK_WARN << "Error reading gold standard point";
  }
}


//-----------------------------------------------------------------------------
std::istream& operator>> (std::istream &is, GoldStandardPoint &GSP )
{
  std::string line;
  if ( std::getline(is,line) )
  {
    std::stringstream linestream(line);
    bool parseSuccess;
    parseSuccess = linestream >> GSP.m_FrameNumber >> GSP.m_Index >> GSP.m_Point.x >> GSP.m_Point.y;
    if ( parseSuccess )
    {
      return is;
    }
    else
    {
      std::stringstream linestream2(line);
      parseSuccess = linestream2 >> GSP.m_FrameNumber >> GSP.m_Point.x >> GSP.m_Point.y;
      if ( parseSuccess )
      {
        GSP.m_Index = -1;
      }
      else
      {
        MITK_WARN << "Error reading gold standard point";
      }
    }
  }
  else
  {
    MITK_WARN << "Error reading gold standard point";
  }
  return is;
}


//-----------------------------------------------------------------------------
bool operator< (const  GoldStandardPoint &GSP1, const GoldStandardPoint &GSP2 )
{
  if ( GSP1.m_FrameNumber == GSP2.m_FrameNumber )
  {
    return GSP1.m_Index < GSP2.m_Index;
  }
  else
  {
    return GSP1.m_FrameNumber < GSP2.m_FrameNumber;
  }
}

//-----------------------------------------------------------------------------
bool operator< (const PickedObject &po1, const PickedObject &po2 )
{
  //by frame number first
  if ( po1.m_FrameNumber == po2.m_FrameNumber )
  {
    //left before right
    if ( po1.m_Channel == po2.m_Channel )
    {
      if ( po1.m_IsLine == po2.m_IsLine )
      {
        return po1.m_Id < po2.m_Id;
      }
      else
      {
        //points before lines
        return ( ! po1.m_IsLine );
      }
    }
    else
    {
      if ( po1.m_Channel == "left" )
      {
        assert ( po2.m_Channel == "right" );
        return true;
      }
      else
      {
        assert ( (po1.m_Channel == "right")  && ( po2.m_Channel == "left") );
        return false;
      }
    }
  }
  else
  {
    return po1.m_FrameNumber < po2.m_FrameNumber;
  }
}



//-----------------------------------------------------------------------------
WorldPoint::WorldPoint()
: m_Point ( cv::Point3d(std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()) )
, m_Scalar (cv::Scalar(255,0,0))
{}


//-----------------------------------------------------------------------------
WorldPoint::WorldPoint(cv::Point3d point, cv::Scalar scalar)
: m_Point ( point )
, m_Scalar ( scalar )
{}


//-----------------------------------------------------------------------------
WorldPoint::WorldPoint(cv::Point3d point)
: m_Point ( point )
, m_Scalar (cv::Scalar(255,0,0))
{}


//-----------------------------------------------------------------------------
WorldPointsWithTimingError::WorldPointsWithTimingError()
: m_Points ()
, m_TimingError ( 0 )
{}


//-----------------------------------------------------------------------------
WorldPointsWithTimingError::WorldPointsWithTimingError(std::vector <mitk::WorldPoint> points,
    long long timingError)
: m_Points ( points )
, m_TimingError ( timingError )
{}


//-----------------------------------------------------------------------------
WorldPointsWithTimingError::WorldPointsWithTimingError(std::vector <mitk::WorldPoint> points)
: m_Points ( points )
, m_TimingError (0)
{}


//-----------------------------------------------------------------------------
ProjectedPointPair::ProjectedPointPair()
: m_Left (cv::Point2d(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()))
, m_Right (cv::Point2d(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()))
{}


//-----------------------------------------------------------------------------
ProjectedPointPair::ProjectedPointPair(cv::Point2d left , cv::Point2d right)
: m_Left (left)
, m_Right (right)
{}


//-----------------------------------------------------------------------------
bool ProjectedPointPair::LeftNaNOrInf()
{
  if ( (boost::math::isnan(m_Left.x)) || 
      (boost::math::isnan(m_Left.y)) || 
      (boost::math::isinf(m_Left.x)) || 
      (boost::math::isinf(m_Left.y)))
  {
    return true;
  }
  else
  {
    return false;
  }    
}


//-----------------------------------------------------------------------------
bool WorldPoint::IsNaN()
{
  if ( (boost::math::isnan(m_Point.x)) ||
      (boost::math::isnan(m_Point.y)) ||
      (boost::math::isnan(m_Point.z)) )
  {
    return true;
  }
  else
  {
    return false;
  }    
}


//-----------------------------------------------------------------------------
bool ProjectedPointPair::RightNaNOrInf()
{
  if ( (boost::math::isnan(m_Right.x)) ||
      (boost::math::isnan(m_Right.y)) ||
      (boost::math::isinf(m_Right.x)) ||
      (boost::math::isinf(m_Right.y)))
  {
    return true;
  }
  else
  {
    return false;
  }    
}


//-----------------------------------------------------------------------------
ProjectedPointPairsWithTimingError::ProjectedPointPairsWithTimingError()
: m_Points()
, m_TimingError(0)
{}


//-----------------------------------------------------------------------------
ProjectedPointPairsWithTimingError::ProjectedPointPairsWithTimingError(
    std::vector <mitk::ProjectedPointPair> points, long long timingError)
: m_Points(points)
, m_TimingError(timingError)
{}


//-----------------------------------------------------------------------------
ProjectedPointPairsWithTimingError::ProjectedPointPairsWithTimingError(
    std::vector <mitk::ProjectedPointPair> points)
: m_Points(points)
, m_TimingError(0)
{}


//-----------------------------------------------------------------------------
VideoFrame::VideoFrame()
{}


//-----------------------------------------------------------------------------
VideoFrame::VideoFrame(cv::VideoCapture* capture , std::ifstream* frameMapLogFile)
{
  if ( ! capture )
  {
    mitkThrow() << "mitk::VideoFrame, passed null video capture.";
    return;
  }
  if ( ! frameMapLogFile )
  {
    mitkThrow() << "mitk::VideoFrame, passed null frame map log file.";
    return;
  }
  bool success = capture->read(m_VideoData);
  if ( ! success )
  {
    mitkThrow() << "mitk::VideoFrame, error reading video file";
    return;
  }
  
  std::string line;
  bool ok = std::getline (*frameMapLogFile, line);
  if ( ! ok )
  {
    mitkThrow() << "mitk::VideoFrame, error getting line from frame map log file";
    return;
  }

  while ( line[0] == '#' )
  {
    ok = std::getline (*frameMapLogFile, line);
    if ( ! ok )
    {
      mitkThrow() << "mitk::VideoFrame, error getting line from frame map log file while skipping comments";
      return;
    }
  }
  
  std::stringstream linestream(line);
  bool parseSuccess = linestream >> m_FrameNumber >> m_SequenceNumber >> m_Channel >> m_TimeStamp;
  if ( ! parseSuccess )
  {
    mitkThrow() << "mitk::VideoFrame, error parsing line from frame map log file";
    return;
  }

  if ( m_Channel == 0 )
  {
    m_Left = true;
  }
  else
  {
    m_Left = false;
  }
  return;
}


//-----------------------------------------------------------------------------
bool VideoFrame::WriteToFile ( std::string prefix )
{
  std::string filename;
  if ( m_Left ) 
  {
    filename = prefix + boost::lexical_cast<std::string>(m_TimeStamp) + "_left.bmp";
  }
  else
  {
    filename = prefix + boost::lexical_cast<std::string>(m_TimeStamp) + "_right.bmp";
  }
  return cv::imwrite( filename, m_VideoData);
}


//-----------------------------------------------------------------------------
void VideoFrame::OutputVideoInformation (cv::VideoCapture * capture)
{
   //output types capture and matrix types
   //
   MITK_INFO << "Video Capture: Frame Width : " << capture->get(CV_CAP_PROP_FRAME_WIDTH);
   MITK_INFO << "Video Capture: Frame Height : " << capture->get(CV_CAP_PROP_FRAME_HEIGHT);


}

//-----------------------------------------------------------------------------
PickedObject::PickedObject()
: m_Id (-1)
, m_IsLine (false)
, m_FrameNumber(0)
, m_TimeStamp(0)
, m_Channel("")
, m_Scalar (cv::Scalar(255,0,0))
{
}

//-----------------------------------------------------------------------------
PickedObject::PickedObject(std::string channel, unsigned int framenumber, unsigned long long timestamp)
: m_Id (-1)
, m_IsLine (false)
, m_FrameNumber(framenumber)
, m_TimeStamp(timestamp)
, m_Channel(channel)
, m_Scalar (cv::Scalar(255,0,0))
{
}

//-----------------------------------------------------------------------------
PickedObject::PickedObject(const GoldStandardPoint& gsp, const unsigned long long & timestamp)
: m_Id (gsp.m_Index)
, m_IsLine (false)
, m_FrameNumber(gsp.m_FrameNumber)
, m_TimeStamp(timestamp)
, m_Channel("")
, m_Scalar (cv::Scalar(255,0,0))
{
  m_Points.push_back(cv::Point3d ( gsp.m_Point.x, gsp.m_Point.y, 0.0 ));
}

//-----------------------------------------------------------------------------
PickedObject::~PickedObject()
{}

//-----------------------------------------------------------------------------
bool PickedObject::HeadersMatch(const PickedObject& otherPickedObject, const long long& allowableTimingError) const
{
  long long timingError;
  if ( m_TimeStamp > otherPickedObject.m_TimeStamp )
  {
    timingError = m_TimeStamp - otherPickedObject.m_TimeStamp;
  }
  else
  {
    timingError = otherPickedObject.m_TimeStamp - m_TimeStamp;
  }

  if ( ( m_Channel ==  otherPickedObject.m_Channel ) &&  
       ( m_IsLine == otherPickedObject.m_IsLine ) &&
       ( m_FrameNumber == otherPickedObject.m_FrameNumber ) &&
       ( ( m_Id == otherPickedObject.m_Id ) || ( otherPickedObject.m_Id == -1 ) ) &&
       ( timingError < allowableTimingError ) )
  {
    return true;
  }
  else
  {
    return false;
  }
}

//-----------------------------------------------------------------------------
PickedObject PickedObject::CopyByHeader() const
{
  PickedObject new_po;
  new_po.m_TimeStamp = m_TimeStamp;
  new_po.m_Channel = m_Channel;
  new_po.m_IsLine = m_IsLine;
  new_po.m_FrameNumber = m_FrameNumber;
  new_po.m_Id = m_Id;

  return new_po;
}


//-----------------------------------------------------------------------------
double PickedObject::DistanceTo(const PickedObject& otherPickedObject , cv::Point3d& deltas,  const long long& allowableTimingError) const
{
  if ( ! otherPickedObject.HeadersMatch (*this) )
  {
    return std::numeric_limits<double>::infinity();
  }
  if ( m_IsLine )
  {
    unsigned int splineOrder = 1;
    return mitk::DistanceBetweenTwoSplines ( m_Points, otherPickedObject.m_Points, splineOrder , &deltas );
  }
  else
  {
    return mitk::DistanceBetweenTwoPoints ( m_Points[0], otherPickedObject.m_Points[0], &deltas );
  }
}

//-----------------------------------------------------------------------------
PickedPointList::PickedPointList()
: m_InLineMode (false)
, m_InOrderedMode (false)
, m_IsModified (false)
, m_XScale (1.0)
, m_YScale (1.0)
{
}

//-----------------------------------------------------------------------------
PickedPointList::~PickedPointList()
{}

//-----------------------------------------------------------------------------
PickedPointList::Pointer PickedPointList::CopyByHeader()
{
  mitk::PickedPointList::Pointer newPL = mitk::PickedPointList::New();
   
  newPL->m_InLineMode = m_InLineMode;
  newPL->m_InOrderedMode = m_InOrderedMode;
  newPL->m_IsModified = true;
  newPL->m_XScale = m_XScale;
  newPL->m_YScale = m_YScale;
  
  newPL->m_TimeStamp = m_TimeStamp;
  newPL->m_FrameNumber = m_FrameNumber;
  newPL->m_Channel = m_Channel;
  return newPL;
}


//-----------------------------------------------------------------------------
void PickedPointList::PutOut (std::ofstream& os )
{
  mitk::SavePickedObjects ( m_PickedObjects, os );
}

//-----------------------------------------------------------------------------
std::vector < mitk::PickedObject > PickedPointList::GetPickedObjects  () const 
{
  return m_PickedObjects;
}

//-----------------------------------------------------------------------------
void PickedPointList::SetPickedObjects (const std::vector < mitk::PickedObject >& objects )
{
  m_PickedObjects = objects;
}

//-----------------------------------------------------------------------------
unsigned int  PickedPointList::GetListSize () const
{
  return m_PickedObjects.size();
}

//-----------------------------------------------------------------------------
void PickedPointList::ClearList ()
{
  m_PickedObjects.clear();
}

//-----------------------------------------------------------------------------
void PickedPointList::AnnotateImage(cv::Mat& image)
{
  for ( int i = 0 ; i < m_PickedObjects.size() ; i ++ )
  {
    std::string number;
    if ( m_PickedObjects[i].m_Id == -1 )
    {
      number = "#";
    }
    else
    {
      number = boost::lexical_cast<std::string>(m_PickedObjects[i].m_Id);
    }
    if ( ! m_PickedObjects[i].m_IsLine )
    {
      assert ( m_PickedObjects[i].m_Points.size() <= 1 );
      for ( unsigned int j = 0 ; j < m_PickedObjects[i].m_Points.size() ; j ++ )
      {
        cv::Point2i point = mitk::Point3dToPoint2i(m_PickedObjects[i].m_Points[j]);
        cv::putText(image,number,point,0,1.0,cv::Scalar(255,255,255));
        cv::circle(image, point,5,cv::Scalar(255,255,255),1,1);
      }
    }
    else
    {
      if ( m_PickedObjects[i].m_Points.size() > 0 )
      {
        for ( unsigned int j = 0 ; j < m_PickedObjects[i].m_Points.size() ; j ++ )
        {
          if ( j == 0 )
          {
            cv::Point2i point = mitk::Point3dToPoint2i(m_PickedObjects[i].m_Points[j]);
            cv::putText(image,number,point,0,1.0,cv::Scalar(255,255,255));
            cv::circle(image, point,5,cv::Scalar(255,255,255),1,1);
          }
          else
          {
            cv::Point2i point = mitk::Point3dToPoint2i(m_PickedObjects[i].m_Points[j]);
            cv::Point2i point2 = mitk::Point3dToPoint2i(m_PickedObjects[i].m_Points[j-1]);
            cv::line(image,  point, point2, cv::Scalar(255,255,255));
          }
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
cv::Mat PickedPointList::CreateMaskImage(const cv::Mat& image)
{
  cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
  
  std::vector < std::vector < cv::Point2i > > contours;
  for ( int i = 0 ; i < m_PickedObjects.size() ; i ++ )
  {
    if ( m_PickedObjects[i].m_IsLine )
    {
      if ( m_PickedObjects[i].m_Points.size() > 0 )
      {
        std::vector < cv::Point2i > points;
        
        for ( unsigned int j = 0 ; j <  m_PickedObjects[i].m_Points.size() ; j ++ )
        {
          points.push_back(mitk::Point3dToPoint2i(m_PickedObjects[i].m_Points[j]));
        }
        contours.push_back( points );
      }
    }
  }
  if ( contours.size() > 0 ) 
  {
    cv::drawContours (mask, contours , -1, cv::Scalar(255), CV_FILLED);
  }
  return mask;
}


//-----------------------------------------------------------------------------
int PickedPointList::GetNextAvailableID( bool ForLine )
{
  int lastPoint = -1 ;
  for ( unsigned int i = 0 ; i < m_PickedObjects.size() ; i ++ )
  {
    if ( (! ForLine ) && (! m_PickedObjects[i].m_IsLine) )
    {
      if ( m_PickedObjects[i].m_Id > lastPoint )
      {
        lastPoint = m_PickedObjects[i].m_Id;
      }
    }
    if ( ( ForLine ) && (m_PickedObjects[i].m_IsLine) )
    {
      if ( m_PickedObjects[i].m_Id > lastPoint )
      {
        lastPoint = m_PickedObjects[i].m_Id;
      }
    }
  }
  return lastPoint+1;
}

//-----------------------------------------------------------------------------
void PickedPointList::SetInOrderedMode(const bool& mode)
{
  if ( m_InOrderedMode == mode )
  {
    return;
  }
  m_InOrderedMode = mode;
  if ( m_InLineMode )
  {
    if ( ! m_InOrderedMode )
    {
      m_PickedObjects.back().m_Id = -1;
    }
    else
    {
      int pointID=this->GetNextAvailableID(true);
      m_PickedObjects.back().m_Id = pointID;
    }
    m_IsModified = true;
  }
}

//-----------------------------------------------------------------------------
void PickedPointList::SetInLineMode(const bool& mode)
{
  if ( m_InLineMode == mode )
  {
    return;
  }
  m_InLineMode = mode;
  if ( m_InLineMode )
  {
    int pointID = -1;
    PickedObject pickedObject(m_Channel, m_FrameNumber, m_TimeStamp);
    pickedObject.m_IsLine = true;

    if ( m_InOrderedMode )
    {
      pointID = this->GetNextAvailableID(true);
    }
    pickedObject.m_Id = pointID;

    m_PickedObjects.push_back(pickedObject);
  }
}

//-----------------------------------------------------------------------------
bool PickedPointList::GetIsModified()
{
  bool state = m_IsModified;
  m_IsModified = false;
  return state;
}

//-----------------------------------------------------------------------------
unsigned int PickedPointList::AddPoint(const cv::Point2i& point)
{
  cv::Point3d myPoint = cv::Point3d (static_cast<double>(point.x) * m_XScale,static_cast<double>(point.y) * m_YScale,0.0);
  if ( m_InLineMode )
  {
    if ( m_PickedObjects.back().m_IsLine )
    {
      m_PickedObjects.back().m_Points.push_back(myPoint);
      MITK_INFO << "Added a point to line " << m_PickedObjects.back().m_Id;
    }
    else
    {
      int pointID=this->GetNextAvailableID(true);
      PickedObject pickedObject(m_Channel, m_FrameNumber, m_TimeStamp);
      pickedObject.m_IsLine = true;
      pickedObject.m_Id = pointID;
      pickedObject.m_Points.push_back(myPoint);

      m_PickedObjects.push_back(pickedObject);
      MITK_INFO << "Created new line at " << m_PickedObjects.back().m_Id;
    }
  }
  else
  {
    if ( m_InOrderedMode )
    {
      int pointID=this->GetNextAvailableID(false);
      PickedObject pickedObject(m_Channel, m_FrameNumber, m_TimeStamp);
      pickedObject.m_IsLine = false;
      pickedObject.m_Id = pointID;
      pickedObject.m_Points.push_back(myPoint);

      m_PickedObjects.push_back(pickedObject);
      MITK_INFO << "Picked ordered point " << pointID << " , " <<  myPoint;
    }
    else
    {
      PickedObject pickedObject(m_Channel, m_FrameNumber, m_TimeStamp);
      pickedObject.m_IsLine = false;
      pickedObject.m_Id = -1;
      pickedObject.m_Points.push_back(myPoint);
      m_PickedObjects.push_back(pickedObject);

      MITK_INFO << "Picked unordered point, " <<  myPoint;
    }
  }
  m_IsModified=true;
  return m_PickedObjects.size();
}

//-----------------------------------------------------------------------------
unsigned int PickedPointList::AddPoint(const cv::Point3d& point, cv::Scalar scalar)
{
  if ( m_InLineMode )
  {
    if ( m_PickedObjects.back().m_IsLine )
    {
      m_PickedObjects.back().m_Points.push_back(point);
      m_PickedObjects.back().m_Scalar = scalar;

      MITK_INFO << "Added a point to line " << m_PickedObjects.back().m_Id;
    }
    else
    {
      int pointID=this->GetNextAvailableID(true);
      PickedObject pickedObject(m_Channel, m_FrameNumber, m_TimeStamp);
      pickedObject.m_IsLine = true;
      pickedObject.m_Id = pointID;
      pickedObject.m_Scalar = scalar;
      pickedObject.m_Points.push_back(point);

      m_PickedObjects.push_back(pickedObject);
      MITK_INFO << "Created new line at " << m_PickedObjects.back().m_Id;
    }
  }
  else
  {
    if ( m_InOrderedMode )
    {
      int pointID=this->GetNextAvailableID(false);
      PickedObject pickedObject(m_Channel, m_FrameNumber, m_TimeStamp);
      pickedObject.m_IsLine = false;
      pickedObject.m_Id = pointID;
      pickedObject.m_Scalar = scalar;
      pickedObject.m_Points.push_back(point);

      m_PickedObjects.push_back(pickedObject);
      MITK_INFO << "Picked ordered point " << pointID << " , " <<  point;
    }
    else
    {
      PickedObject pickedObject(m_Channel, m_FrameNumber, m_TimeStamp);
      pickedObject.m_IsLine = false;
      pickedObject.m_Id = -1;
      pickedObject.m_Scalar = scalar;
      pickedObject.m_Points.push_back(point);
      m_PickedObjects.push_back(pickedObject);

      MITK_INFO << "Picked unordered point, " <<  point;
    }
  }
  m_IsModified=true;
  return m_PickedObjects.size();
}


//-----------------------------------------------------------------------------
unsigned int PickedPointList::RemoveLastPoint()
{
  if ( m_PickedObjects.size() != 0 )
  {
    if ( m_InLineMode )
    {
      std::vector<PickedObject>::iterator it = m_PickedObjects.end() -1 ;
      while ( it >= m_PickedObjects.begin() )
      {
        if ( it->m_IsLine )
        {
          MITK_INFO << "found a line at " << it->m_Id;
          if ( it->m_Points.size() > 0 )
          {
            it->m_Points.pop_back();
            MITK_INFO << "Removed last point from line " << it->m_Id;
          }
          else
          {
            MITK_INFO << "Removed line " << it->m_Id;
            m_PickedObjects.erase(it);
          }
          break;
        }
        else
        {
          MITK_INFO << it->m_Id << " is not a line";
        }
        it -- ;
      }
    }
    else
    {
      MITK_INFO << "Removing last point";
      std::vector<PickedObject>::iterator it = m_PickedObjects.end() - 1;
      while ( it >= m_PickedObjects.begin() )
      {
        if ( ! (it->m_IsLine) )
        {
          MITK_INFO << "found a point";
          m_PickedObjects.erase(it);
          break;
        }
        it -- ;
      }
    }
    m_IsModified=true;
  }
  return m_PickedObjects.size();
}

//-----------------------------------------------------------------------------
unsigned int PickedPointList::SkipOrderedPoint()
{
  if ( ! m_InOrderedMode )
  {
    if ( ! m_InLineMode )
    {
      return m_PickedObjects.size();
    }
    else 
    {
      PickedObject pickedObject(m_Channel, m_FrameNumber, m_TimeStamp);
      pickedObject.m_IsLine = true;
      pickedObject.m_Id = -1;

      m_PickedObjects.push_back(pickedObject);

      MITK_INFO << "Skipped unordered line";
    }
    return m_PickedObjects.size();
  }

  if ( ! m_InLineMode )
  {
    int pointID=this->GetNextAvailableID(false);
    PickedObject pickedObject(m_Channel, m_FrameNumber, m_TimeStamp);
    pickedObject.m_IsLine = false;
    pickedObject.m_Id = pointID;

    m_PickedObjects.push_back(pickedObject);

    MITK_INFO << "Skipped ordered point " << pointID;
  }
  else
  {
    int pointID=this->GetNextAvailableID(true);

    PickedObject pickedObject(m_Channel, m_FrameNumber, m_TimeStamp);
    pickedObject.m_IsLine = true;
    pickedObject.m_Id = pointID;

    m_PickedObjects.push_back(pickedObject);

    MITK_INFO << "Skipped ordered line " << pointID-1;
  }
  return m_PickedObjects.size();
}

//-----------------------------------------------------------------------------
void PointPickingCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
  PickedPointList* out = static_cast<PickedPointList*>(userdata);
  if  ( flags == cv::EVENT_FLAG_LBUTTON  )
  {
    out->AddPoint (cv::Point2i ( x,y));
    return;
  }
  else if  ( flags == cv::EVENT_FLAG_RBUTTON )
  {
    out->RemoveLastPoint();
    return;
  }
  else if  ( ( event == cv::EVENT_MBUTTONDOWN ) ||
      ( flags == cv::EVENT_FLAG_CTRLKEY + cv::EVENT_FLAG_LBUTTON ) ||
      ( flags == cv::EVENT_FLAG_CTRLKEY + cv::EVENT_FLAG_RBUTTON ) )
  {
    out->SkipOrderedPoint();
    return;
  }
}

//-----------------------------------------------------------------------------
cv::Point2i Point3dToPoint2i (const cv::Point3d& point)
{
  if ( fabs ( point.z ) > 1e-6 )
  {
    mitkThrow() << "Attempted to cast point3d to point2i with non zero z";
  }
  return cv::Point2i ( boost::math::round(point.x), boost::math::round(point.y) );
}
} // end namespace
