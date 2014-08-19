/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackingMatrixTimeStamps.h"
#include <algorithm>
#include <mitkExceptionMacro.h>
#include <sstream>

namespace mitk {


//---------------------------------------------------------------------------
void TrackingMatrixTimeStamps::Sort()
{
  std::sort(m_TimeStamps.begin(), m_TimeStamps.end());
}


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
bool TrackingMatrixTimeStamps::GetBoundingTimeStamps(const unsigned long long& input,
                                                     unsigned long long& before,
                                                     unsigned long long& after,
                                                     double& proportion
                                                    )
{
  bool isValid = false;
  before = 0;             // So that even if user fails to check return code,
  after = 0;              // they will notice a lack of timestamps.
  proportion = 0;

  if (m_TimeStamps.size() == 0)
  {
    return isValid;
  }

  std::vector<unsigned long long>::iterator iter = std::lower_bound (m_TimeStamps.begin() , m_TimeStamps.end(), input);

  if (iter == m_TimeStamps.end())
  {
    --iter;
    before = *iter;
    return isValid;
  }

  if (iter == m_TimeStamps.begin())
  {
    after = *iter;
    return isValid;
  }

  if (*iter == input)
  {
    before = input;
    after = input;
    proportion = 0;
    isValid = true;
    return isValid;
  }

  after = *iter;
  --iter;
  before = *iter;
  proportion = static_cast<double>(input - before)/static_cast<double>(after-before);
  isValid = true;

  return isValid;
}


//---------------------------------------------------------------------------
unsigned long long TrackingMatrixTimeStamps::GetNearestTimeStamp (const unsigned long long& timestamp, long long *delta)
{
  unsigned long long before = 0;
  unsigned long long after = 0;
  unsigned long long result = 0;
  double proportion = 0;
  bool isValid = false;
  long long diff = 0;

  isValid = this->GetBoundingTimeStamps(timestamp, before, after, proportion);

  if (!isValid)
  {
    if (before != 0)
    {
      // Not a valid interval, but before is still the nearest we can find.
      result = before;
    }
    else if (after != 0)
    {
      // Not a valid interval, but after is still the nearest we can find.
      result = after;
    }
    else
    {
      result = 0;
    }
  }
  else // isValid == true
  {
    if (proportion == 0)
    {
      // exact match, can pick either.
      result = before;
    }
    else if (proportion > 0 && proportion <= 1)
    {
      long long deltaUpper = after - timestamp;
      long long deltaLower = timestamp - before;

      if ( abs(deltaLower) <= abs(deltaUpper) )
      {
        result = before;
      }
      else
      {
        result = after;
      }
    }
    else
    {
      std::ostringstream errorMessage;
      errorMessage << "Programming error:isValid=" << isValid << ", proportion=" << proportion << ", giving up :-(" << std::endl;
      mitkThrow() << errorMessage.str();
    }
  }

  if (result == 0)
  {
    diff = 0;
  }
  else
  {
    diff = timestamp - result;
  }

  // User provided a non-null output variable, so now we write to it.
  if ( delta != NULL )
  {
    *delta = diff;
  }

  // Then return the timestamp.
  return result;
}

} // end namespace
