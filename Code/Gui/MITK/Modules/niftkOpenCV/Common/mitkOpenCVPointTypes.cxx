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
#include <numeric>
#include <algorithm>
#include <functional>
#include <mitkMathsUtils.h>
#include <mitkExceptionMacro.h>

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

} // end namespace
