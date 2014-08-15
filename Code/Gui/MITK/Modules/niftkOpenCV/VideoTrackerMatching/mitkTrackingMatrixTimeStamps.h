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
 * \brief Helper class that contains a vector of timestamps, that is assumed to be strictly increasing.
 */
class NIFTKOPENCV_EXPORT TrackingMatrixTimeStamps
{
public:
  std::vector<unsigned long long> m_TimeStamps;

  /**
   * \brief Retrieves the timestamps before and after a given point.
   * \param[Input] input timestamp, normally in nano-seconds since Unix Epoch (UTC).
   * \param[Output] before timestamp, normally in nano-seconds since Unix Epoch (UTC).
   * \param[Output] after timestamp, normally in nano-seconds since Unix Epoch (UTC).
   * \return the fraction, from [0 to 1] of what proportion the input timestamp is of the interval between before and after.
   * If m_TimeStamps is empty, will copy input to both before and after.
   */
  double GetBoundingTimeStamps(const unsigned long long& input,
                               unsigned long long& before,
                               unsigned long long& after
                              );

  /**
   * \brief Retrieves the closest timestamp, and if delta is non-null, will populate with the error.
   * \param[Input] input timestamp, normally in nano-seconds since Unix Epoch (UTC).
   * \param[Output] delta the error, i.e. the number of nanoseconds between the requested timestamp, and the returned timestamp.
   * \return timestamp
   */
  unsigned long long GetNearestTimeStamp (const unsigned long long& input , long long * delta = NULL );
};

} // end namespace

#endif
