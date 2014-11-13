/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkMIDASHelper.h"

namespace itk
{

//-----------------------------------------------------------------------------
int GetAxisFromOrientationString(const std::string& orientationString, const itk::Orientation& orientation)
{
  int outputAxis = -1;

  if (orientationString != "UNKNOWN")
  {
    for (int i = 0; i < 3; i++)
    {
      if (orientation == itk::ORIENTATION_AXIAL && (orientationString[i] == 'S' || orientationString[i] == 'I'))
      {
        outputAxis = i;
        break;
      }

      if (orientation == itk::ORIENTATION_CORONAL && (orientationString[i] == 'A' || orientationString[i] == 'P'))
      {
        outputAxis = i;
        break;
      }

      if (orientation == itk::ORIENTATION_SAGITTAL && (orientationString[i] == 'L' || orientationString[i] == 'R'))
      {
        outputAxis = i;
        break;
      }
    }
  }
  return outputAxis;
}


//-----------------------------------------------------------------------------
int GetUpDirection(const std::string& orientationString, const int& axisOfInterest)
{
  int upDirection = 0;

  // NOTE: ITK convention is that an image that goes from
  // Left to Right in X, Posterior to Anterior in Y and Inferior to Superior in Z
  // is called an LPI, whereas in Nifti speak, that would be RAS.

  if (orientationString != "UNKNOWN" && axisOfInterest != -1)
  {
    char direction = orientationString[axisOfInterest];
    if (direction == 'A' || direction == 'S' || direction == 'R')
    {
      upDirection = -1;
    }
    else if (direction == 'P' || direction == 'I' || direction == 'L')
    {
      upDirection = 1;
    }
  }
  return upDirection;
}

//-----------------------------------------------------------------------------
std::string GetMajorAxisFromPatientRelativeDirectionCosine(double x,double y,double z)
{
  double obliquityThresholdCosineValue = 0.8;

  std::string axis;

  std::string orientationX = x < 0 ? "R" : "L";
  std::string orientationY = y < 0 ? "A" : "P";
  std::string orientationZ = z < 0 ? "F" : "H";

  double absX = fabs(x);
  double absY = fabs(y);
  double absZ = fabs(z);

  // The tests here really don't need to check the other dimensions,
  // just the threshold, since the sum of the squares should be == 1.0
  // but just in case ...

  if (absX>obliquityThresholdCosineValue && absX>absY && absX>absZ)
  {
    axis=orientationX;
  }
  else if (absY>obliquityThresholdCosineValue && absY>absX && absY>absZ)
  {
    axis=orientationY;
  }
  else if (absZ>obliquityThresholdCosineValue && absZ>absX && absZ>absY)
  {
    axis=orientationZ;
  }

  return axis;
}

} // end namespace
