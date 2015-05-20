/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkProjectionErrors.h"
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/special_functions/round.hpp>
#include <numeric>
#include <algorithm>
#include <functional>
#include <mitkMathsUtils.h>
#include <mitkExceptionMacro.h>
#include <string>
#include <fstream>

namespace mitk {

//-----------------------------------------------------------------------------
mitk::PickedObject ProjectionErrorCalculator::FindNearestScreenPoint ( mitk::PickedObject GSPoint, std::string channel , double* minRatio, unsigned int* index)
{
  unsigned int matches =0;
  mitk::PickedObject matchingObject;
  if ( GSPoint.m_Id != -1 )
  {
    if ( index != NULL ) 
    {
      *index = GSPoint.m_Id;
    }
    if ( minRatio != NULL ) 
    {
      *minRatio = m_AllowablePointMatchingRatio + 1.0;
    }

    for ( std::vector<mitk::PickedObject>::iterator it = m_ProjectedPoints.begin() ; it < m_ProjectedPoints.end() ; it++ )
    {
      if ( it->HeadersMatch (GSPoint) )
      {
        matchingObject = *it;
        matches++;
      }
    }
  }
  else
  {
    if ( m_ClassifierProjectedPoints.size() != m_ProjectedPoints.size() )
    {
      mitkThrow() << "mitkProjectionErrors::FindNearestPoint classifier and projected point list sizes differ: " << 
      m_ClassifierProjectedPoints.size() << " != " <<  m_ProjectedPoints.size() ;
    }

    std::vector < mitk::PickedObject > pointVector;
    for ( std::vector<mitk::PickedObject>::iterator it = m_ClassifierProjectedPoints.begin() ; it < m_ClassifierProjectedPoints.rnd() ; it ++ )
    {
      if ( it->HeadersMatch (GSPoint) )
      {
        pointVector.push_back ( *it );
      }
    }
    unsigned int myIndex;
    if ( ! boost::math::isinf(mitk::FindNearestPoint( GSPoint.m_Point , pointVector ,minRatio, &myIndex ).x))
    {
      if ( index != NULL ) 
      {
        *index = myIndex;
      }
      if ( left ) 
      {
        return m_ProjectedPoints[GSPoint.m_FrameNumber].m_Points[myIndex].m_Left;
      }
      else
      {
        return m_ProjectedPoints[GSPoint.m_FrameNumber].m_Points[myIndex].m_Right;
      }
    }
    else
    {
      return cv::Point2d ( std::numeric_limits<double>::infinity() , std::numeric_limits<double>::infinity() ) ;
    }
  }
  if ( matches > 1 )
  {
    mitkThrow() << "mitkProjectionErrors::FindNearestScreenPoint found multiple matches: " << matches;
  }
  if ( matches == 0 )
  {
    mitkThrow() << "mitkProjectionErrors::FindNearestScreenPoint found no matches: " << matches;
  }
}

} // end namespace
