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
} // end namespace
