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
#include <numeric>
#include <algorithm>
#include <functional>
#include <mitkMathsUtils.h>
#include <mitkExceptionMacro.h>
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
: id (-1)
, isLine (false)
{
}

//-----------------------------------------------------------------------------
PickedObject::~PickedObject()
{}

//-----------------------------------------------------------------------------
PickedPointList::PickedPointList()
: m_InLineMode (false)
, m_InOrderedMode (false)
, m_IsModified (false)
{
}

//-----------------------------------------------------------------------------
PickedPointList::~PickedPointList()
{}

//-----------------------------------------------------------------------------
void PickedPointList::PutOut (std::ofstream& os )
{
  os << "<frame>" <<  m_FrameNumber << "</frame>" << std::endl;
  os << "<channel>" << m_Channel <<"</channel>" << std::endl;
      
  for ( int i = 0 ; i < m_PickedObjects.size(); i ++ )
  {
    if ( m_PickedObjects[i].isLine )
    {
      if ( m_PickedObjects[i].points.size() > 0 )
      {
        os << "<line>" << std::endl;
        os << "<id>" << m_PickedObjects[i].id << "</id>" << std::endl;
        os << "<coordinates>" <<std::endl;
        for ( unsigned int j = 0 ; j < m_PickedObjects[i].points.size() ; j ++ )
        {
          os << m_PickedObjects[i].points[j];
        }
        os << std::endl << "</coordinates>" <<std::endl;
        os << "</line>" << std::endl;
      }
    }
    else
    {
      if ( m_PickedObjects[i].points.size() > 0 )
      {
        os << "<point>" << std::endl;
        os << "<id>" << m_PickedObjects[i].id << "</id>" << std::endl;
        os << "<coordinates>" <<std::endl;
        for ( unsigned int j = 0 ; j < m_PickedObjects[i].points.size() ; j ++ )
        {
          os << m_PickedObjects[i].points[j];
        }
        os << std::endl << "</coordinates>" <<std::endl;
        os << "</point>" << std::endl;
      }
    }
  }
}

//-----------------------------------------------------------------------------
void PickedPointList::AnnotateImage(cv::Mat& image)
{
  for ( int i = 0 ; i < m_PickedObjects.size() ; i ++ )
  {
    std::string number;
    if ( m_PickedObjects[i].id == -1 )
    {
      number = "#";
    }
    else
    {
      number = boost::lexical_cast<std::string>(m_PickedObjects[i].id);
    }
    if ( ! m_PickedObjects[i].isLine )
    {
      assert ( m_PickedObjects[i].points.size() <= 1 );
      for ( unsigned int j = 0 ; j < m_PickedObjects[i].points.size() ; j ++ )
      {
        cv::putText(image,number,m_PickedObjects[i].points[j],0,1.0,cv::Scalar(255,255,255));
        cv::circle(image, m_PickedObjects[i].points[j],5,cv::Scalar(255,255,255),1,1);
      }
    }
    else
    {
      if ( m_PickedObjects[i].points.size() > 0 )
      {
        for ( unsigned int j = 0 ; j < m_PickedObjects[i].points.size() ; j ++ )
        {
          if ( j == 0 )
          {
            cv::putText(image,number,m_PickedObjects[i].points[j],0,1.0,cv::Scalar(255,255,255));
            cv::circle(image, m_PickedObjects[i].points[j],5,cv::Scalar(255,255,255),1,1);
          }
          else
          {
            cv::line(image,  m_PickedObjects[i].points[j],  m_PickedObjects[i].points[j-1], cv::Scalar(255,255,255));
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
    if ( m_PickedObjects[i].isLine )
    {
      if ( m_PickedObjects[i].points.size() > 0 )
      {
        contours.push_back( m_PickedObjects[i].points );
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
    if ( (! ForLine ) && (! m_PickedObjects[i].isLine) )
    {
      if ( m_PickedObjects[i].id > lastPoint )
      {
        lastPoint = m_PickedObjects[i].id;
      }
    }
    if ( ( ForLine ) && (m_PickedObjects[i].isLine) )
    {
      if ( m_PickedObjects[i].id > lastPoint )
      {
        lastPoint = m_PickedObjects[i].id;
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
      m_PickedObjects.back().id = -1;
    }
    else
    {
      int pointID=this->GetNextAvailableID(true);
      m_PickedObjects.back().id = pointID;
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
    PickedObject pickedObject;
    pickedObject.isLine = true;

    if ( m_InOrderedMode )
    {
      pointID = this->GetNextAvailableID(true);
    }
    pickedObject.id = pointID;

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
  if ( m_InLineMode )
  {
    if (  m_PickedObjects.back().isLine )
    {
      m_PickedObjects.back().points.push_back(point);
      MITK_INFO << "Added a point to line " << m_PickedObjects.back().id;
    }
    else
    {
      int pointID=this->GetNextAvailableID(true);
      PickedObject pickedObject;
      pickedObject.isLine = true;
      pickedObject.id = pointID;
      pickedObject.points.push_back(point);

      m_PickedObjects.push_back(pickedObject);
      MITK_INFO << "Created new line at " << m_PickedObjects.back().id;
    }
  }
  else
  {
    if ( m_InOrderedMode )
    {
      int pointID=this->GetNextAvailableID(false);
      PickedObject pickedObject;
      pickedObject.isLine = false;
      pickedObject.id = pointID;
      pickedObject.points.push_back(point);

      m_PickedObjects.push_back(pickedObject);
      MITK_INFO << "Picked ordered point " << pointID << " , " <<  point;
    }
    else
    {
      PickedObject pickedObject;
      pickedObject.isLine = false;
      pickedObject.id = -1;
      pickedObject.points.push_back(point);
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
        if ( it->isLine )
        {
          MITK_INFO << "found a line at " << it->id;
          if ( it->points.size() > 0 )
          {
            it->points.pop_back();
            MITK_INFO << "Removed last point from line " << it->id;
          }
          else
          {
            MITK_INFO << "Removed line " << it->id;
            m_PickedObjects.erase(it);
          }
          break;
        }
        else
        {
          MITK_INFO << it->id << " is not a line";
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
        if ( ! (it->isLine) )
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
    return m_PickedObjects.size();
  }

  if ( ! m_InLineMode )
  {
    int pointID=this->GetNextAvailableID(false);
    PickedObject pickedObject;
    pickedObject.isLine = false;
    pickedObject.id = pointID;

    m_PickedObjects.push_back(pickedObject);

    MITK_INFO << "Skipped ordered point " << pointID;
  }
  else
  {
    int pointID=this->GetNextAvailableID(true);

    PickedObject pickedObject;
    pickedObject.isLine = true;
    pickedObject.id = pointID;

    m_PickedObjects.push_back(pickedObject);

    MITK_INFO << "Skipped ordered line " << pointID-1;
  }
  return m_PickedObjects.size();
}

//-----------------------------------------------------------------------------
void PointPickingCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
  PickedPointList* out = static_cast<PickedPointList*>(userdata);
  if  ( event == cv::EVENT_LBUTTONDOWN )
  {
    out->AddPoint (cv::Point2i ( x,y));
  }
  else if  ( event == cv::EVENT_RBUTTONDOWN )
  {
    out->RemoveLastPoint();
  }
  else if  ( event == cv::EVENT_MBUTTONDOWN )
  {
    out->SkipOrderedPoint();
  }
}


} // end namespace
