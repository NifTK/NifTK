/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTrackingMatrixTimeStamps_h
#define mitkTrackingMatrixTimeStamps_h

#include "niftkOpenCVExports.h"
#include <cv.h>

namespace mitk
{

/**
 * \class TrackingMatrixTimeStamps
 * \brief Helper class that contains a vector of timestamps, that are assumed to be strictly increasing.
 *
 * See also mitkTrackingMatrixTimeStampsTest.cxx.
 */
class NIFTKOPENCV_EXPORT TrackingMatrixTimeStamps
{
public:

  typedef unsigned long long TimeStamp;

  /**
   * \brief Simply adds a TimeStamp to the list.
   */
  void Insert(const TimeStamp& timeStamp);

  /**
   * \brief Gets the number of timestamps.
   */
  std::vector<TrackingMatrixTimeStamps::TimeStamp>::size_type GetSize() const;

  /**
   * \brief Returns the time stamp at a given frameNumber.
   */
  TimeStamp GetTimeStamp(std::vector<TrackingMatrixTimeStamps::TimeStamp>::size_type frameNumber) const;

  /**
   * \brief Empties the list.
   */
  void Clear();

  /**
   * \brief Sorts the list.
   */
  void Sort();

  /**
   * \brief Given a timeStamp in nanoseconds, will search the list for the corresponding array index, returning -1 if not found.
   * \param[in] timeStamp in nano-seconds since Unix Epoch (UTC).
   * \return vector index number or -1 if not found.
   */
  int GetFrameNumber(const TimeStamp& timeStamp);

  /**
   * \brief Retrieves the timestamps before and after a given point.
   *
   * \param[in] timeStamp in nano-seconds since Unix Epoch (UTC).
   * \param[out] before timestamp in nano-seconds since Unix Epoch (UTC).
   * \param[out] after timestamp in nano-seconds since Unix Epoch (UTC).
   * \param[out] proportion the fraction from [0 to 1] of what proportion the input timestamp is of the interval between before and after.
   * \return true if valid bounding interval and false otherwise.
   *
   * Additional Spec:
   *   - If no timestamps, before = 0, after = 0, proportion = 0, return false.
   *   - If input > all items in list, before = last item in list (i.e. nearest lower bound), after = 0, (i.e. invalid), proportion = 0, return false.
   *   - If input < all items in list, after = first item in list (i.e. nearest upper bound), before = 0, (i.e invalid), proportion = 0, return false.
   *   - If input exactly matches item in list, before = input, after = input, proportion = 0, return true;
   *   - Otherwise, before is timestamp just before, after is timestamp just after the given input, proportion is linear interpolation between timestamps, return true;
   */
  bool GetBoundingTimeStamps(const TimeStamp& timeStamp,
                               TimeStamp& before,
                               TimeStamp& after,
                               double& proportion
                              );

  /**
   * \brief Retrieves the closest timestamp, and if delta is non-null, will populate with the error.
   *
   * \param[in] timeStamp in nano-seconds since Unix Epoch (UTC).
   * \param[out] delta i.e. the number of nanoseconds between the requested timestamp, and the returned timestamp.
   * \return nearest timestamp
   *
   * Additional Spec:
   *   - If no timestamps, return 0, delta = 0 if provided
   */
  TimeStamp GetNearestTimeStamp (const TimeStamp& timeStamp , long long * delta = NULL );

private:
  std::vector<TimeStamp> m_TimeStamps;

};

} // end namespace

#endif
