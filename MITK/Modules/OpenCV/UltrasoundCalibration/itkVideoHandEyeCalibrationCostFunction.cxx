/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkVideoHandEyeCalibrationCostFunction.h"
#include <mitkOpenCVMaths.h>

namespace itk {

//-----------------------------------------------------------------------------
VideoHandEyeCalibrationCostFunction::VideoHandEyeCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
VideoHandEyeCalibrationCostFunction::~VideoHandEyeCalibrationCostFunction()
{
}



//-----------------------------------------------------------------------------
cv::Matx44d VideoHandEyeCalibrationCostFunction::GetCalibrationTransformation(const ParametersType & parameters) const
{
  cv::Matx44d rigid = this->GetRigidTransformation(parameters);
  return rigid;
}

//-----------------------------------------------------------------------------
} // end namespace
