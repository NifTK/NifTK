/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPointsAndNormalsBasedRegistration.h"
#include <limits>
#include <mitkPointUtils.h>
#include <mitkLiuLeastSquaresWithNormalsRegistrationWrapper.h>
#include <mitkArunLeastSquaresPointRegistrationWrapper.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

const bool mitk::PointsAndNormalsBasedRegistration::DEFAULT_USE_POINT_ID_TO_MATCH(true);

namespace mitk
{

//-----------------------------------------------------------------------------
PointsAndNormalsBasedRegistration::PointsAndNormalsBasedRegistration()
: m_UsePointIDToMatchPoints(DEFAULT_USE_POINT_ID_TO_MATCH)
{
}


//-----------------------------------------------------------------------------
PointsAndNormalsBasedRegistration::~PointsAndNormalsBasedRegistration()
{
}


//-----------------------------------------------------------------------------
bool PointsAndNormalsBasedRegistration::Update(
    const mitk::PointSet::Pointer fixedPointSet,
    const mitk::PointSet::Pointer movingPointSet,
    const mitk::PointSet::Pointer fixedNormals,
    const mitk::PointSet::Pointer movingNormals,
    vtkMatrix4x4& outputTransform,
    double& fiducialRegistrationError) const
{
  assert(fixedPointSet);
  assert(movingPointSet);
  assert(fixedNormals);
  assert(movingNormals);

  bool isSuccessful = false;

  fiducialRegistrationError = std::numeric_limits<double>::max();
  outputTransform.Identity();

  mitk::PointSet::Pointer filteredFixedPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer filteredFixedNormals = mitk::PointSet::New();
  mitk::PointSet::Pointer filteredMovingPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer filteredMovingNormals = mitk::PointSet::New();

  mitk::PointSet* fixedPoints = fixedPointSet;
  mitk::PointSet* movingPoints = movingPointSet;
  mitk::PointSet* fixedNorms = fixedNormals;
  mitk::PointSet* movingNorms = movingNormals;

  if (m_UsePointIDToMatchPoints)
  {
    int numberOfFilteredPoints = mitk::FilterMatchingPoints(*fixedPointSet,
                                                            *movingPointSet,
                                                            *filteredFixedPoints,
                                                            *filteredMovingPoints
                                                            );

    int numberOfFilteredNormals = mitk::FilterMatchingPoints(*fixedNormals,
                                                             *movingNormals,
                                                             *filteredFixedNormals,
                                                             *filteredMovingNormals
                                                            );
    if (numberOfFilteredPoints >= 2)
    {
      fixedPoints = filteredFixedPoints;
      movingPoints = filteredMovingPoints;
    }
    else
    {
      MITK_DEBUG << "mitk::PointsAndNormalsBasedRegistration: filteredFixedPoints size=" << filteredFixedPoints->GetSize() << ", filteredMovingPoints size=" << filteredMovingPoints->GetSize() << ", abandoning use of filtered data sets.";
      return isSuccessful;
    }

    if (numberOfFilteredNormals >= 2)
    {
      fixedNorms = filteredFixedNormals;
      movingNorms = filteredMovingNormals;
    }
    else
    {
      MITK_DEBUG << "mitk::PointsAndNormalsBasedRegistration: filteredFixedNormals size=" << filteredFixedNormals->GetSize() << ", filteredMovingNormals size=" << filteredMovingNormals->GetSize() << ", abandoning use of filtered data sets.";
      return isSuccessful;
    }

    if (numberOfFilteredPoints != numberOfFilteredNormals)
    {
      MITK_DEBUG << "mitk::PointsAndNormalsBasedRegistration: numberOfFilteredPoints=" << numberOfFilteredPoints << ", numberOfFilteredNormals=" << numberOfFilteredNormals << ", abandoning use of filtered data sets.";
      return isSuccessful;
    }
  }

  if (fixedPoints->GetSize() < 2 || movingPoints->GetSize() < 2)
  {
    MITK_DEBUG << "mitk::PointsAndNormalsBasedRegistration:: fixedPoints size=" << fixedPoints->GetSize() << ", movingPoints size=" << movingPoints->GetSize() << ", abandoning point based registration";
    return isSuccessful;
  }

  // Two pass registration.

  // First create augmented data set.
  mitk::PointSet::Pointer augmentedFixedPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer augmentedMovingPoints = mitk::PointSet::New();

  mitk::PointSet::DataType* itkPointSet = fixedPoints->GetPointSet(0);
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType fixedPoint, fixedNormal, movingPoint, movingNormal, additionalFixedPoint, additionalMovingPoint;

  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();

    fixedPoint = fixedPoints->GetPoint(pointID);
    fixedNormal = fixedNorms->GetPoint(pointID);
    movingPoint = movingPoints->GetPoint(pointID);
    movingNormal = movingNorms->GetPoint(pointID);

    augmentedFixedPoints->InsertPoint(pointID, fixedPoint);
    augmentedMovingPoints->InsertPoint(pointID, movingPoint);

    double scaleFactor = 10;
    int pointIDOffset = 1024;
    for (int i = 0; i < 3; i++)
    {
      additionalFixedPoint[i] = fixedPoint[i] + scaleFactor*fixedNormal[i];
      additionalMovingPoint[i] = movingPoint[i] + scaleFactor*movingNormal[i];
    }
    augmentedFixedPoints->InsertPoint(pointID+pointIDOffset, additionalFixedPoint);
    augmentedMovingPoints->InsertPoint(pointID+pointIDOffset, additionalMovingPoint);
  }

  // Then do 'normal', SVD, point based registration.
  vtkSmartPointer<vtkMatrix4x4> arunMatrix = vtkMatrix4x4::New();
  arunMatrix->Identity();
  double arunFudicialRegistrationError = std::numeric_limits<double>::max();
  mitk::ArunLeastSquaresPointRegistrationWrapper::Pointer arunRegistration = mitk::ArunLeastSquaresPointRegistrationWrapper::New();
  isSuccessful = arunRegistration->Update(augmentedFixedPoints, augmentedMovingPoints, *arunMatrix, arunFudicialRegistrationError);
  if (!isSuccessful)
  {
    MITK_ERROR << "mitk::PointsAndNormalsBasedRegistration: First point based SVD failed" << std::endl;
    return isSuccessful;
  }

  // Transform moving points and normals according to arunMatrix.
  mitk::PointSet::Pointer transformedMovingPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer transformedMovingNormals = mitk::PointSet::New();
  itkPointSet = movingPoints->GetPointSet(0);
  points = itkPointSet->GetPoints();
  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();

    // Note: reusing some variable names from above.
    movingPoint = movingPoints->GetPoint(pointID);
    movingNormal = movingNorms->GetPoint(pointID);

    mitk::TransformPointByVtkMatrix(arunMatrix, false, movingPoint);
    mitk::TransformPointByVtkMatrix(arunMatrix, true, movingNormal);

    transformedMovingPoints->InsertPoint(pointID, movingPoint);
    transformedMovingNormals->InsertPoint(pointID, movingNormal);
  }

  // Now do SVD, point based registration, encorporating surface normals.
  vtkSmartPointer<vtkMatrix4x4> liuMatrix = vtkMatrix4x4::New();
  liuMatrix->Identity();
  double liuFudicialRegistrationError = std::numeric_limits<double>::max();
  mitk::LiuLeastSquaresWithNormalsRegistrationWrapper::Pointer liuRegistration = mitk::LiuLeastSquaresWithNormalsRegistrationWrapper::New();
  isSuccessful = liuRegistration->Update(fixedPoints, fixedNorms, transformedMovingPoints, transformedMovingNormals, *liuMatrix, liuFudicialRegistrationError);

  if (!isSuccessful)
  {
    MITK_ERROR << "mitk::PointsAndNormalsBasedRegistration: Second point based SVD failed" << std::endl;
    return isSuccessful;
  }

  MITK_INFO << "mitk::PointsAndNormalsBasedRegistration: FRE=" << arunFudicialRegistrationError << ", then " << liuFudicialRegistrationError << std::endl;

  // Combine the two transformations, and report the final error.
  if (liuFudicialRegistrationError < arunFudicialRegistrationError)
  {
    vtkMatrix4x4::Multiply4x4(liuMatrix, arunMatrix, &outputTransform);
    fiducialRegistrationError = liuFudicialRegistrationError;
  }
  else
  {
    // Fallback to just Arun method, using Normals as fake points.
    outputTransform.DeepCopy(arunMatrix);
    fiducialRegistrationError = arunFudicialRegistrationError;
    MITK_WARN << "mitk::PointsAndNormalsBasedRegistration: Falling back to just Arun's method" << std::endl;
  }

  return isSuccessful;
}

} // end namespace

