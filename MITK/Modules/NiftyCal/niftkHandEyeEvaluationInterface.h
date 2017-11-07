/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkHandeyeEvaluationInterface_h
#define niftkHandeyeEvaluationInterface_h

#include "niftkNiftyCalExports.h"
#include <string>

namespace niftk
{

NIFTKNIFTYCAL_EXPORT int EvaluateHandeyeFromPoints(const std::string& trackingDir,
                                                   const std::string& pointsDir,
                                                   const std::string& modelFile,
                                                   const std::string& intrinsicsFile,
                                                   const std::string& handeyeFile,
                                                   const std::string& registrationFile,
                                                   double &rmsError
                                                  );
} // end namespace

#endif
