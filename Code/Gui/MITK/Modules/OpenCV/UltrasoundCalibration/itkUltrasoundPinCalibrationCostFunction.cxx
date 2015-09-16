/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkUltrasoundPinCalibrationCostFunction.h"
#include <mitkOpenCVMaths.h>

namespace itk {

//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::UltrasoundPinCalibrationCostFunction()
: m_OptimiseScaleFactors(true)
{
  m_ScaleFactors[0] = 1;
  m_ScaleFactors[1] = 1;
}


//-----------------------------------------------------------------------------
UltrasoundPinCalibrationCostFunction::~UltrasoundPinCalibrationCostFunction()
{
}


//-----------------------------------------------------------------------------
cv::Matx44d UltrasoundPinCalibrationCostFunction::GetScalingTransformation(const ParametersType & parameters) const
{
  cv::Matx44d scaleFactors;
  mitk::MakeIdentity(scaleFactors);

  if (parameters.GetSize() == 11 && this->GetOptimiseScaleFactors())
  {
    scaleFactors(0,0) = parameters[9];
    scaleFactors(1,1) = parameters[10];
  }
  else
  {
    scaleFactors(0,0) = m_ScaleFactors[0];
    scaleFactors(1,1) = m_ScaleFactors[1];
  }

  return scaleFactors;
}


//-----------------------------------------------------------------------------
cv::Matx44d UltrasoundPinCalibrationCostFunction::GetCalibrationTransformation(const ParametersType & parameters) const
{
  cv::Matx44d rigid = this->GetRigidTransformation(parameters);
  cv::Matx44d scaling = this->GetScalingTransformation(parameters);
  cv::Matx44d similarity = rigid * scaling;
  return similarity;
}

//-----------------------------------------------------------------------------
} // end namespace
