/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkLuiLeastSquaresWithNormalsRegistration.h"
#include <mitkOpenCVMaths.h>

namespace mitk {

//-----------------------------------------------------------------------------
LuiLeastSquaresWithNormalsRegistration::LuiLeastSquaresWithNormalsRegistration()
{
}


//-----------------------------------------------------------------------------
LuiLeastSquaresWithNormalsRegistration::~LuiLeastSquaresWithNormalsRegistration()
{
}


//-----------------------------------------------------------------------------
bool LuiLeastSquaresWithNormalsRegistration::Update(const std::vector<cv::Point3d>& fixedPoints,
                                                    const std::vector<cv::Point3d>& fixedNormals,
                                                    const std::vector<cv::Point3d>& movingPoints,
                                                    const std::vector<cv::Point3d>& movingNormals,
                                                    cv::Matx44d& outputMatrix,
                                                    double &fiducialRegistrationError)
{
  bool success = false;
  unsigned int numberOfPoints = fixedPoints.size();

  return success;
}


//-----------------------------------------------------------------------------
} // end namespace





