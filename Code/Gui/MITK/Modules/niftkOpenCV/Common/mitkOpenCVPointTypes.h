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
#include <cv.h>

/**
 * \file mitkOpenCVPointTypes.h
 * \brief Derived point types to contain data for projection and analysis
 */
namespace mitk {


class NIFTKOPENCV_EXPORT GoldStandardPoint
{
  /**
  * \class contains the gold standard points
  * consisting of the frame number, the point and optionally the point index
  */
  public:
    GoldStandardPoint();
    GoldStandardPoint(unsigned int , int, cv::Point2d);
    GoldStandardPoint(std::istream& is);
    unsigned int m_FrameNumber;
    int m_Index;
    cv::Point2d  m_Point;

    /** 
    * \brief an input operator
    */
    friend std::istream& operator>> (std::istream& is, const GoldStandardPoint& gsp );

    friend bool operator < ( const GoldStandardPoint &GSP1 , const GoldStandardPoint &GSP2);
};

class NIFTKOPENCV_EXPORT WorldPoint
{
  /**
   * \class contains a point in 3D and a corresponding scalar value
   */
   public:
     WorldPoint();
     WorldPoint(cv::Point3d, cv::Scalar);
     WorldPoint(cv::Point3d);

     cv::Point3d  m_Point;
     cv::Scalar   m_Scalar;
};

class NIFTKOPENCV_EXPORT WorldPointsWithTimingError
{
  /**
   * \class contains a vector of world points and a corresponding long long timing error
   */
   public:
     WorldPointsWithTimingError();
     WorldPointsWithTimingError(std::vector <mitk::WorldPoint>, long long);
     WorldPointsWithTimingError(std::vector <mitk::WorldPoint>);

     std::vector <mitk::WorldPoint> m_Points;
     long long                      m_TimingError;
};


class NIFTKOPENCV_EXPORT ProjectedPointPair
{
  /**
   * \class contains a left and right projected point
   */

  ProjectedPointPair();
  ProjectedPointPair(cv::Point2d, cv::Point2d);

  cv::Point2d m_Left;
  cv::Point2d m_Right;
};

class NIFTKOPENCV_EXPORT ProjectedPointPairsWithTimingError
{
  /**
   * \class contains a vector of left and right projected points and a timing error
   */
  public:
    ProjectedPointPairsWithTimingError();
    ProjectedPointPairsWithTimingError(std::vector <mitk::ProjectedPointPair>, long long);
    ProjectedPointPairsWithTimingError(std::vector <mitk::ProjectedPointPair>);

    std::vector <mitk::ProjectedPointPair> m_Points;
    long long                              m_TimingError; 
};

} // end namespace

#endif



