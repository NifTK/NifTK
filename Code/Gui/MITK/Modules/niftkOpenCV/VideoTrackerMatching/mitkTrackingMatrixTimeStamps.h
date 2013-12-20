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

class NIFTKOPENCV_EXPORT TrackingMatrixTimeStamps
{
public:
  std::vector<unsigned long long> m_TimeStamps;  
  unsigned long long GetNearestTimeStamp (const unsigned long long& timestamp , long long * delta = NULL );
};

} // end namespace

#endif
