/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTrackingMatrices_h
#define mitkTrackingMatrices_h

#include "niftkOpenCVExports.h"
#include <cv.h>

namespace mitk
{

class NIFTKOPENCV_EXPORT TrackingMatrices
{
public:
  std::vector<cv::Mat>   m_TrackingMatrices;
  std::vector<long long> m_TimingErrors;
};

} // end namespace

#endif
