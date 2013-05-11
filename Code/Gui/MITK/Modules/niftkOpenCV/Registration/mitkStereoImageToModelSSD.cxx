/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoImageToModelSSD.h"
#include "mitkRegistrationHelper.h"
#include <mitkCameraCalibrationFacade.h>

namespace mitk
{

//-----------------------------------------------------------------------------
StereoImageToModelSSD::StereoImageToModelSSD()
{
}


//-----------------------------------------------------------------------------
StereoImageToModelSSD::~StereoImageToModelSSD()
{
}


//-----------------------------------------------------------------------------
StereoImageToModelMetric::MeasureType StereoImageToModelSSD::CalculateCost(
    const CvMat& transformed3DPoints,
    const CvMat& transformed3DNormals,
    const CvMat& weights,
    const CvMat& transformed2DPointsLeft,
    const CvMat& transformed2DPointsRight,
    const ParametersType &parameters
    ) const
{
  StereoImageToModelMetric::MeasureType currentCost = 0;

  bool successful = false;
  double diff = 0;

  int channels = this->m_InputLeftImage->nChannels;

  float *leftValues = new float [channels];
  float *rightValues = new float [channels];

  int numberOfPoints = transformed3DPoints.rows;
  int numberOfSuccessfulPoints = 0;

  for (int i = 0; i < numberOfPoints; i++)
  {
    successful = this->GetImageValues(
        CV_MAT_ELEM(transformed2DPointsLeft, float, i, 0),
        CV_MAT_ELEM(transformed2DPointsLeft, float, i, 1),
        CV_MAT_ELEM(transformed2DPointsRight, float, i, 0),
        CV_MAT_ELEM(transformed2DPointsRight, float, i, 1),
        leftValues,
        rightValues
        );
    if (successful)
    {
      diff = 0;
      for (int j = 0; j < channels; j++)
      {
        //diff += ((leftValues[j] - rightValues[j])*(leftValues[j] - rightValues[j])*CV_MAT_ELEM(weights, float, i, 0));
        diff += fabs(leftValues[j] - rightValues[j]);
      }
      currentCost += diff;
      numberOfSuccessfulPoints++;
    }
  }

  if (numberOfSuccessfulPoints > 0)
  {
    currentCost /= numberOfSuccessfulPoints;
  }

  delete [] leftValues;
  delete [] rightValues;

  return currentCost;
}

//-----------------------------------------------------------------------------
} // end namespace mitk

