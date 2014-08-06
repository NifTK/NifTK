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
     WorldPoint(std::istream& is);

     cv::Point3d  m_Point;
     cv::Scalar   m_Scalar;
};

} // end namespace

#endif



