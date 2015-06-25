/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkProjectionErrors.h"
#include "mitkOpenCVMaths.h"
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
ProjectionErrorCalculator::ProjectionErrorCalculator ()
{}

//-----------------------------------------------------------------------------
ProjectionErrorCalculator::~ProjectionErrorCalculator ()
{}

//-----------------------------------------------------------------------------
mitk::PickedObject ProjectionErrorCalculator::FindNearestScreenPoint ( mitk::PickedObject GSPoint, std::string channel , double* minRatio)
{
  unsigned int matches =0;
  mitk::PickedObject matchingObject;
  if ( GSPoint.m_Id != -1 )
  {
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
    matchingObject = mitk::FindNearestPoint( GSPoint , m_ClassifierProjectedPoints ,minRatio );
    if ( matchingObject.m_Points.size () != 0)
    {
      matches = 1;
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
  return matchingObject;
}

} // end namespace
