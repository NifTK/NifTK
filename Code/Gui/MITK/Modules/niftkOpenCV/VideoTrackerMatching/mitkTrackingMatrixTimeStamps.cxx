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
int TrackingMatrixTimeStamps::GetFrameNumber(const unsigned long long& timeStamp)
{
  int result = -1;

  for (unsigned long int i = 0; i < m_TimeStamps.size(); i++)
  {
    if (m_TimeStamps[i] == timeStamp)
    {
      result = i;
      break;
    }
  }

  return result;
}


//---------------------------------------------------------------------------
double TrackingMatrixTimeStamps::GetBoundingTimeStamps(const unsigned long long& input,
                                                       unsigned long long& before,
                                                       unsigned long long& after
                                                      )
{
  double proportion = 0;

  if (m_TimeStamps.size() == 0)
  {
    before = input;
    after = input;
    return proportion;
  }

  std::vector<unsigned long long>::iterator upper = std::upper_bound (m_TimeStamps.begin() , m_TimeStamps.end(), input);
  std::vector<unsigned long long>::iterator lower = std::lower_bound (m_TimeStamps.begin() , m_TimeStamps.end(), input);

  if (upper == m_TimeStamps.end())
  {
    --upper;
  }
  if (lower == m_TimeStamps.end())
  {
    --lower;
  }

  if (*lower >= input && lower != m_TimeStamps.begin())
  {
    lower--;
  }

  before = *lower;
  after = *upper;

  if (upper != lower)
  {
    proportion = static_cast<double>(input - before)/static_cast<double>(after-before);
  }

  return proportion;
}


//---------------------------------------------------------------------------
unsigned long long TrackingMatrixTimeStamps::GetNearestTimeStamp (const unsigned long long& timestamp, long long *error)
{
  unsigned long long before, after;
  this->GetBoundingTimeStamps(timestamp, before, after);

  long long deltaUpper = after - timestamp;
  long long deltaLower = timestamp - before;

  unsigned long long returnValue = timestamp;

  long long delta;
  if ( deltaLower == 0 ) 
  {
    returnValue = before;
    delta = 0;
  }
  else
  {
    if ( abs(deltaLower) <= abs(deltaUpper) )
    {
      returnValue = before;
      delta = (long long) timestamp - before;
    }
    else
    {
      returnValue = after;
      delta = (long long) timestamp - after;
    }
  }

  // User provided a non-null output variable, so now we write to it.
  if ( error != NULL )
  {
    *error = delta;
  }

  // Then return the timestamp.
  return returnValue;
}

} // end namespace
