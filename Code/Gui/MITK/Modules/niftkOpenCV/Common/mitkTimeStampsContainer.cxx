/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTimeStampsContainer.h"
#include <algorithm>
#include <mitkExceptionMacro.h>
#include <sstream>
#include <vector>
#include <cassert>

namespace mitk {


//---------------------------------------------------------------------------
void TimeStampsContainer::Insert(const TimeStamp& timeStamp)
{
  m_TimeStamps.push_back(timeStamp);
}


//---------------------------------------------------------------------------
TimeStampsContainer::TimeStamp TimeStampsContainer::GetTimeStamp(std::vector<TimeStampsContainer::TimeStamp>::size_type frameNumber) const
{
  assert(frameNumber >= 0);
  assert(frameNumber < m_TimeStamps.size());
  return m_TimeStamps[frameNumber];
}


//---------------------------------------------------------------------------
void TimeStampsContainer::Sort()
{
  std::sort(m_TimeStamps.begin(), m_TimeStamps.end());
}


//---------------------------------------------------------------------------
void TimeStampsContainer::Clear()
{
  m_TimeStamps.clear();
}


//---------------------------------------------------------------------------
std::vector<TimeStampsContainer::TimeStamp>::size_type TimeStampsContainer::GetSize() const
{
  return m_TimeStamps.size();
}


//---------------------------------------------------------------------------
std::vector<TimeStampsContainer::TimeStamp>::size_type TimeStampsContainer::GetFrameNumber(const TimeStamp& timeStamp) const
{
  std::vector<TimeStampsContainer::TimeStamp>::size_type result = -1;
  std::vector<TimeStampsContainer::TimeStamp>::size_type i;

  for (i = 0; i < m_TimeStamps.size(); i++)
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
bool TimeStampsContainer::GetBoundingTimeStamps(const TimeStamp& input,
                                                     TimeStamp& before,
                                                     TimeStamp& after,
                                                     double& proportion
                                                    ) const
{
  bool isValid = false;
  before = 0;             // So that even if user fails to check return code,
  after = 0;              // they will notice a lack of timestamps.
  proportion = 0;

  if (m_TimeStamps.size() == 0)
  {
    return isValid;
  }

  if (input < *(m_TimeStamps.begin()))
  {
    after = *(m_TimeStamps.begin());
    return isValid;
  }

  if (input == *(m_TimeStamps.begin()))
  {
    before = input;
    after = input;
    proportion = 0;
    isValid = true;
    return isValid;
  }

  // This assumes that the value tested for is within the range.
  std::vector<unsigned long long>::const_iterator iter = std::lower_bound (m_TimeStamps.begin() , m_TimeStamps.end(), input);

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
unsigned long long TimeStampsContainer::GetNearestTimeStamp (const TimeStamp& timestamp, long long *delta) const
{
  TimeStamp before = 0;
  TimeStamp after = 0;
  TimeStamp result = 0;
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

      assert ( ( deltaUpper > 0 ) && (deltaLower > 0));

      if ( deltaLower <= deltaUpper )
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
