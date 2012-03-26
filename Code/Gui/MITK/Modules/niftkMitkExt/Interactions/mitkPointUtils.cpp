/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkPointUtils.h"
#include "mitkCommon.h"

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

bool mitk::AreDifferent(const mitk::Point3D& a, const mitk::Point3D& b)
{
  bool areDifferent = false;

  for (int i = 0; i < 3; i++)
  {
    if (fabs(a[i] - b[i]) > 0.01)
    {
      areDifferent = true;
    }
  }

  return areDifferent;
}

float mitk::GetSquaredDistanceBetweenPoints(const mitk::Point3D& a, const mitk::Point3D& b)
{
    double distance = 0;

    for (int i = 0; i < 3; i++)
    {
      distance += (a[i] - b[i])*(a[i] - b[i]);
    }

    return distance;
}

void mitk::GetDifference(const mitk::Point3D& a, const mitk::Point3D& b, mitk::Point3D& output)
{
  for (int i = 0; i < 3; i++)
  {
    output[i] = a[i] - b[i];
  }
}

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
