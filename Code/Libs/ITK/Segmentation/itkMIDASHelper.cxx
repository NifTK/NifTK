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

} // end namespace
