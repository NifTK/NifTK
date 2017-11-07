/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkTimingCalibrationInterface.h"
#include <niftkNiftyCalTypes.h>
#include <cv.h>
#include <vector>

namespace niftk
{

//-----------------------------------------------------------------------------
int EvaluateHandeyeFromPoints(const std::string& trackingDir,
                              const std::string& pointsDir,
                              const std::string& modelFile,
                              const std::string& intrinsicsFile,
                              const std::string& handeyeFile,
                              const std::string& registrationFile,
                              double &rmsError
                             )
{
  std::cout << "Matt, EvaluateHandeyeFromPoints" << std::endl;
}

} // end namespace
