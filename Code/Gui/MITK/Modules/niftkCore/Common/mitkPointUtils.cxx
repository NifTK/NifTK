/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPointUtils.h"
#include <mitkCommon.h>

//-----------------------------------------------------------------------------
double mitk::CalculateStepSize(double *spacing)
{
  double stepSize = 0;
  double smallestDimension = std::numeric_limits<double>::max();

  for (int i = 0; i< 3; i++)
  {
    if (spacing[i] < smallestDimension)
    {
      smallestDimension = spacing[i];
    }
  }
  stepSize = smallestDimension / 3.0;
  return stepSize;
}


//-----------------------------------------------------------------------------
bool mitk::AreDifferent(const mitk::Point3D& a, const mitk::Point3D& b)
{
  bool areDifferent = false;

  for (int i = 0; i < 3; i++)
  {
    if (fabs(a[i] - b[i]) > 0.01)
    {
      areDifferent = true;
      break;
    }
  }

  return areDifferent;
}


//-----------------------------------------------------------------------------
float mitk::GetSquaredDistanceBetweenPoints(const mitk::Point3D& a, const mitk::Point3D& b)
{
    double distance = 0;

    for (int i = 0; i < 3; i++)
    {
      distance += (a[i] - b[i])*(a[i] - b[i]);
    }

    return distance;
}


//-----------------------------------------------------------------------------
void mitk::GetDifference(const mitk::Point3D& a, const mitk::Point3D& b, mitk::Point3D& output)
{
  for (int i = 0; i < 3; i++)
  {
    output[i] = a[i] - b[i];
  }
}


//-----------------------------------------------------------------------------
double mitk::Length(mitk::Point3D& vector)
{
  double length = 0;
  for (int i = 0; i < 3; i++)
  {
    length += vector[i]*vector[i];
  }
  if (length > 0)
  {
    length = sqrt(length);
  }
  return length;
}


//-----------------------------------------------------------------------------
void mitk::Normalise(mitk::Point3D& vector)
{
  double length = Length(vector);
  if (length > 0)
  {
    for (int i = 0; i < 3; i++)
    {
      vector[i] /= length;
    }
  }
}


//-----------------------------------------------------------------------------
int mitk::CopyPointSets(const mitk::PointSet& input, mitk::PointSet& output)
{
  output.Clear();
  for (int i = 0; i < input.GetSize(); ++i)
  {
    output.InsertPoint(i, input.GetPoint(i));
  }
  return output.GetSize();
}

