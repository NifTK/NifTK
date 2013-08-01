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


//-----------------------------------------------------------------------------
void mitk::CopyValues(const mitk::Point3D& a, mitk::Point3D& b)
{
  for (int i = 0; i < 3; i++)
  {
    b[i] = a[i];
  }
}


//-----------------------------------------------------------------------------
void mitk::CrossProduct(const mitk::Point3D& a, const mitk::Point3D& b, mitk::Point3D& c)
{
  mitk::Point3D aCopy;
  mitk::Point3D bCopy;
  CopyValues(a, aCopy);
  CopyValues(b, bCopy);
  Normalise(aCopy);
  Normalise(bCopy);

  c[0] = aCopy[1]*bCopy[2] - bCopy[1]*aCopy[2];
  c[1] = -1 * (aCopy[0]*bCopy[2] - bCopy[0]*aCopy[2]);
  c[2] = aCopy[0]*bCopy[1] - bCopy[0]*aCopy[1];
}


//-----------------------------------------------------------------------------
void mitk::ComputeNormalFromPoints(const mitk::Point3D& a, const mitk::Point3D& b, const mitk::Point3D& c, mitk::Point3D& output)
{
  mitk::Point3D aMinusB;
  mitk::Point3D cMinusB;
  GetDifference(a, b, aMinusB);
  GetDifference(c, b, cMinusB);
  CrossProduct(aMinusB, cMinusB, output);
}



//-----------------------------------------------------------------------------
void mitk::TransformPointByVtkMatrix(
    vtkMatrix4x4* matrix,
    const bool& isNormal,
    mitk::Point3D& point
    )
{
  double transformedPoint[4] = {0, 0, 0, 1};

  if(matrix != NULL)
  {
    transformedPoint[0] = point[0];
    transformedPoint[1] = point[1];
    transformedPoint[2] = point[2];
    transformedPoint[3] = 1;

    matrix->MultiplyPoint(transformedPoint, transformedPoint);

    point[0] = transformedPoint[0];
    point[1] = transformedPoint[1];
    point[2] = transformedPoint[2];

    if (isNormal)
    {
      double transformedOrigin[4] = {0, 0, 0, 1};
      matrix->MultiplyPoint(transformedOrigin, transformedOrigin);

      point[0] = point[0] - transformedOrigin[0];
      point[1] = point[1] - transformedOrigin[1];
      point[2] = point[2] - transformedOrigin[2];
    }
  }
  else
  {
    transformedPoint[0] = point[0];
    transformedPoint[1] = point[1];
    transformedPoint[2] = point[2];
  }
}


//-----------------------------------------------------------------------------
int mitk::FilterMatchingPoints(
    const mitk::PointSet& fixedPointsIn,
    const mitk::PointSet& movingPointsIn,
    mitk::PointSet& fixedPointsOut,
    mitk::PointSet& movingPointsOut
    )
{
  int matchedPoints = 0;
  fixedPointsOut.Clear();
  movingPointsOut.Clear();

  mitk::PointSet::DataType* fixedPointSet = fixedPointsIn.GetPointSet(0);
  mitk::PointSet::PointsContainer* fixedPoints = fixedPointSet->GetPoints();
  mitk::PointSet::DataType* movingPointSet = movingPointsIn.GetPointSet(0);
  mitk::PointSet::PointsContainer* movingPoints = movingPointSet->GetPoints();

  mitk::PointSet::PointsIterator fixedPointsIt;
  mitk::PointSet::PointsIterator movingPointsIt;

  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType fixedPoint;
  mitk::PointSet::PointType movingPoint;

  for (fixedPointsIt = fixedPoints->Begin(); fixedPointsIt != fixedPoints->End(); ++fixedPointsIt)
  {
    pointID = fixedPointsIt->Index();
    fixedPoint = fixedPointsIt->Value();

    for (movingPointsIt = movingPoints->Begin(); movingPointsIt != movingPoints->End(); ++movingPointsIt)
    {
      if (movingPointsIt->Index() == pointID)
      {
        movingPoint = movingPointsIt->Value();

        fixedPointsOut.InsertPoint(pointID, fixedPoint);
        movingPointsOut.InsertPoint(pointID, movingPoint);
        matchedPoints++;
      }
    }
  }

  return matchedPoints;
}


