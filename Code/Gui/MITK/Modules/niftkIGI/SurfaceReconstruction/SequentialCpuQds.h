/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkSequentialCpuQds_h
#define mitkSequentialCpuQds_h

#include "niftkIGIExports.h"
#include <opencv2/core/types_c.h>


namespace mitk 
{



class NIFTKIGI_EXPORT SequentialCpuQds
{

public:
  void Process(const IplImage* left, const IplImage* right);
};

} // namespace

#endif // mitkSequentialCpuQds_h
