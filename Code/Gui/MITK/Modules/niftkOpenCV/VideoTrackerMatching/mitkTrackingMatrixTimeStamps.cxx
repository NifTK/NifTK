/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackingMatrixTimeStamps.h"

namespace mitk {

//---------------------------------------------------------------------------
unsigned long long TrackingMatrixTimeStamps::GetNearestTimeStamp (const unsigned long long& timestamp, long long * Delta)
{
  std::vector<unsigned long long>::iterator upper = std::upper_bound (m_TimeStamps.begin() , m_TimeStamps.end(), timestamp);
  std::vector<unsigned long long>::iterator lower = std::lower_bound (m_TimeStamps.begin() , m_TimeStamps.end(), timestamp);

  if (upper == m_TimeStamps.end())
    --upper;
  if (lower == m_TimeStamps.end())
    --lower;

  long long deltaUpper = *upper - timestamp ;
  long long deltaLower = timestamp - *lower ;
  unsigned long long returnValue;
  long long delta;
  if ( deltaLower == 0 ) 
  {
    returnValue = *lower;
    delta = 0;
  }
  else
  {
    if (lower != m_TimeStamps.begin())
      --lower;

    deltaLower = timestamp - *lower;
    if ( abs(deltaLower) < abs(deltaUpper) ) 
    {
      returnValue = *lower;
      delta = (long long) timestamp - *lower;
    }
    else
    {
      returnValue = *upper;
      delta = (long long) timestamp - *upper;
    }
  }

  if ( Delta != NULL ) 
  {
    *Delta = delta;
  }
  return returnValue;
}

} // end namespace
