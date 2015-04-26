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
#include <boost/math/special_functions/fpclassify.hpp>

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
double mitk::GetSquaredDistanceBetweenPoints(const mitk::Point3D& a, const mitk::Point3D& b)
{
    double distance = 0;

    for (int i = 0; i < 3; i++)
    {
      distance += (a[i] - b[i])*(a[i] - b[i]);
    }

    return distance;
}


//-----------------------------------------------------------------------------
double mitk::GetRMSErrorBetweenPoints(
  const mitk::PointSet& fixedPoints, 
  const mitk::PointSet& movingPoints, 
  const mitk::CoordinateAxesData * const transform)
{
  mitk::PointSet::DataType* itkPointSet = movingPoints.GetPointSet();
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType fixedPoint;
  mitk::PointSet::PointType movingPoint;
  mitk::PointSet::PointType transformedMovingPoint;
  
  double rmsError = 0;
  unsigned long int numberOfPointsUsed = 0;
  
  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();
    movingPoint = pIt->Value();
    
    if (fixedPoints.GetPointIfExists(pointID, &fixedPoint))
    {
      if (transform != NULL)
      {
        transformedMovingPoint = transform->MultiplyPoint(movingPoint); 
        rmsError += mitk::GetSquaredDistanceBetweenPoints(fixedPoint, transformedMovingPoint);
      }
      else
      {
        rmsError += mitk::GetSquaredDistanceBetweenPoints(fixedPoint, movingPoint);
      }
      numberOfPointsUsed++;
    }
  }
  if (numberOfPointsUsed > 0)
  {
    rmsError /= static_cast<double>(numberOfPointsUsed);
    rmsError = sqrt(rmsError);      
  }
  else
  {
    rmsError = 0;
  }
  return rmsError;
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
double mitk::FindLargestDistanceBetweenTwoPoints(const mitk::PointSet& input)
{
  double maxSquaredDistance = 0;

  mitk::PointSet::PointsContainer* inputContainer = input.GetPointSet()->GetPoints();
  mitk::PointSet::PointsConstIterator outerIt = inputContainer->Begin();
  mitk::PointSet::PointsConstIterator innerIt = inputContainer->Begin();
  mitk::PointSet::PointsConstIterator iterEnd = inputContainer->End();

  for ( ; outerIt != iterEnd; ++outerIt)
  {
    for ( ; innerIt != iterEnd; ++outerIt)
    {
      double squaredDistance = mitk::GetSquaredDistanceBetweenPoints(outerIt->Value(), innerIt->Value());
      if (squaredDistance > maxSquaredDistance)
      {
        maxSquaredDistance = squaredDistance;
      }
    }
  }
  return sqrt(maxSquaredDistance);
}


//-----------------------------------------------------------------------------
int mitk::CopyPointSets(const mitk::PointSet& input, mitk::PointSet& output)
{
  output.Clear();

  mitk::PointSet::PointsContainer* inputContainer = input.GetPointSet()->GetPoints();
  mitk::PointSet::PointsConstIterator inputIt = inputContainer->Begin();
  mitk::PointSet::PointsConstIterator inputEnd = inputContainer->End();
  for ( ; inputIt != inputEnd; ++inputIt)
  {
    output.InsertPoint(inputIt->Index(), inputIt->Value());
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
  Normalise(output);
}


//-----------------------------------------------------------------------------
void mitk::TransformPointByVtkMatrix(
    const vtkMatrix4x4* matrix,
    const bool& isNormal,
    mitk::Point3D& point
    )
{
  double transformedPoint[4] = {0, 0, 0, 1};
  vtkMatrix4x4* nonConstMatrix = const_cast<vtkMatrix4x4*>(matrix);

  if(nonConstMatrix != NULL)
  {
    transformedPoint[0] = point[0];
    transformedPoint[1] = point[1];
    transformedPoint[2] = point[2];
    transformedPoint[3] = 1;

    nonConstMatrix->MultiplyPoint(transformedPoint, transformedPoint);

    point[0] = transformedPoint[0];
    point[1] = transformedPoint[1];
    point[2] = transformedPoint[2];

    if (isNormal)
    {
      double transformedOrigin[4] = {0, 0, 0, 1};
      nonConstMatrix->MultiplyPoint(transformedOrigin, transformedOrigin);

      point[0] = point[0] - transformedOrigin[0];
      point[1] = point[1] - transformedOrigin[1];
      point[2] = point[2] - transformedOrigin[2];
    }
  }
}


//-----------------------------------------------------------------------------
void mitk::TransformPointsByVtkMatrix(
    const mitk::PointSet& input,
    const vtkMatrix4x4& matrix,
    mitk::PointSet& output
    )
{
  mitk::PointSet::DataType* itkPointSet = input.GetPointSet();
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType point;

  output.Clear();

  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();
    point = pIt->Value();
    mitk::TransformPointByVtkMatrix(&matrix, false, point);
    output.InsertPoint(pointID, point);
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


//-----------------------------------------------------------------------------
int mitk::RemoveNaNPoints(
    const mitk::PointSet& pointsIn,
    mitk::PointSet& pointsOut
    )
{
  int removedPoints = 0;
  pointsOut.Clear();

  mitk::PointSet::DataType* pointSet = pointsIn.GetPointSet(0);
  mitk::PointSet::PointsContainer* points = pointSet->GetPoints();

  mitk::PointSet::PointsIterator pointsIt;

  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType point;

  for (pointsIt = points->Begin(); pointsIt != points->End(); ++pointsIt)
  {
    pointID = pointsIt->Index();
    point = pointsIt->Value();

        
    if ( mitk::CheckForNaNPoint(point) )
    {
      removedPoints++;
    }
    else
    {
      pointsOut.InsertPoint(pointID, point);
    }
  }
  return removedPoints;
}


//-----------------------------------------------------------------------------
bool mitk::CheckForNaNPoint( const mitk::PointSet::PointType& point )
{
  if ( boost::math::isnan( point[0] ) || boost::math::isnan( point[1] ) || boost::math::isnan( point[2] ))
  {
    return true;
  }
  return false;
}


//-----------------------------------------------------------------------------
mitk::Point3D mitk::ComputeCentroid(
    const mitk::PointSet& input
    )
{
  mitk::Point3D average;
  average.Fill(0);

  if (input.GetSize() > 0)
  {
    mitk::PointSet::DataType* pointSet = input.GetPointSet(0);
    mitk::PointSet::PointsContainer* points = pointSet->GetPoints();
    mitk::PointSet::PointsIterator pointsIt;
    mitk::PointSet::PointType point;

    for (pointsIt = points->Begin(); pointsIt != points->End(); ++pointsIt)
    {
      point = pointsIt->Value();
      average[0] += point[0];
      average[1] += point[1];
      average[2] += point[2];
    }

    double numberOfPoints = static_cast<double>(input.GetSize());

    average[0] /= numberOfPoints;
    average[1] /= numberOfPoints;
    average[2] /= numberOfPoints;
  }

  return average;
}


