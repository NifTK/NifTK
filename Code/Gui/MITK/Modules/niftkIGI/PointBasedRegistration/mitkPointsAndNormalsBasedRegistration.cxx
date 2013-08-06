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
const bool mitk::PointsAndNormalsBasedRegistration::DEFAULT_USE_TWO_PHASE(true);
const bool mitk::PointsAndNormalsBasedRegistration::DEFAULT_USE_EXHAUSTIVE_SEARCH(true);

namespace mitk
{

//-----------------------------------------------------------------------------
PointsAndNormalsBasedRegistration::PointsAndNormalsBasedRegistration()
: m_UsePointIDToMatchPoints(DEFAULT_USE_POINT_ID_TO_MATCH)
, m_UseTwoPhase(DEFAULT_USE_TWO_PHASE)
, m_UseExhaustiveSearch(DEFAULT_USE_EXHAUSTIVE_SEARCH)
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

  // Working data.
  mitk::PointSet::Pointer augmentedFixedPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer augmentedMovingPoints = mitk::PointSet::New();

  mitk::PointSet::Pointer transformedMovingPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer transformedMovingNormals = mitk::PointSet::New();

  double arunFudicialRegistrationError = std::numeric_limits<double>::max();
  vtkSmartPointer<vtkMatrix4x4> arunMatrix = vtkMatrix4x4::New();
  arunMatrix->Identity();

  double liuFudicialRegistrationError = std::numeric_limits<double>::max();
  vtkSmartPointer<vtkMatrix4x4> liuMatrix = vtkMatrix4x4::New();
  liuMatrix->Identity();

  mitk::PointSet::DataType* itkPointSet = NULL;
  mitk::PointSet::PointsContainer* points = NULL;
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointsIterator pIt2;
  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointIdentifier pointID2;
  mitk::PointSet::PointType fixedPoint, fixedNormal, movingPoint, movingNormal, additionalFixedPoint, additionalMovingPoint;

  if (m_UseTwoPhase)
  {
    // We 'augment' the point sets, by creating fake points based on surface normals.
    // We then treat these fake points as if they were real points, and do a
    // standard SVD based point registration.
    itkPointSet = fixedPoints->GetPointSet(0);
    points = itkPointSet->GetPoints();

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
    mitk::ArunLeastSquaresPointRegistrationWrapper::Pointer arunRegistration = mitk::ArunLeastSquaresPointRegistrationWrapper::New();
    isSuccessful = arunRegistration->Update(augmentedFixedPoints, augmentedMovingPoints, *arunMatrix, arunFudicialRegistrationError);
    if (!isSuccessful)
    {
      MITK_ERROR << "mitk::PointsAndNormalsBasedRegistration: Arun's' point based SVD failed" << std::endl;
      return isSuccessful;
    }

    MITK_INFO << "mitk::PointsAndNormalsBasedRegistration: Arun's method, FRE=" << arunFudicialRegistrationError << std::endl;
  }

  // Transform moving points and normals according to arunMatrix.
  // Therefore if m_UseTwoPhase is false, arunMatrix will be Identity.

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

  // Now do SVD, point based registration, encorporating surface normals, a la Liu and Fitzpatrick paper.

  mitk::LiuLeastSquaresWithNormalsRegistrationWrapper::Pointer liuRegistration = mitk::LiuLeastSquaresWithNormalsRegistrationWrapper::New();
  isSuccessful = liuRegistration->Update(fixedPoints, fixedNorms, transformedMovingPoints, transformedMovingNormals, *liuMatrix, liuFudicialRegistrationError);

  if (!isSuccessful)
  {
    MITK_ERROR << "mitk::PointsAndNormalsBasedRegistration: Liu's' point based SVD failed" << std::endl;
    return isSuccessful;
  }

  MITK_INFO << "mitk::PointsAndNormalsBasedRegistration: Liu, Fitzpatrick method, FRE=" << liuFudicialRegistrationError << std::endl;

  // Additional exhaustive search for best of 2 points.
  if (m_UseExhaustiveSearch)
  {
    double bestSoFarRegistrationError = std::numeric_limits<double>::max();
    vtkSmartPointer<vtkMatrix4x4> bestSoFarRegistrationMatrix = vtkMatrix4x4::New();
    bestSoFarRegistrationMatrix->Identity();

    mitk::PointSet::PointIdentifier bestSoFarPointID1;
    mitk::PointSet::PointIdentifier bestSoFarPointID2;

    itkPointSet = fixedPoints->GetPointSet(0);
    points = itkPointSet->GetPoints();

    for (pIt = points->Begin(); pIt != points->End(); ++pIt)
    {
      for (pIt2 = pIt, pIt2++; pIt2 != points->End(); ++pIt2)
      {
        pointID = pIt->Index();
        pointID2 = pIt2->Index();

        bool tmpIsSuccessful = false;
        double tmpRegistrationError = std::numeric_limits<double>::max();
        vtkSmartPointer<vtkMatrix4x4> tmpRegistrationMatrix = vtkMatrix4x4::New();
        tmpRegistrationMatrix->Identity();

        mitk::PointSet::Pointer tmpFixedPoints = mitk::PointSet::New();
        mitk::PointSet::Pointer tmpFixedNormals = mitk::PointSet::New();

        mitk::PointSet::Pointer tmpMovingPoints = mitk::PointSet::New();
        mitk::PointSet::Pointer tmpMovingNormals = mitk::PointSet::New();

        tmpFixedPoints->InsertPoint(pointID, fixedPoints->GetPoint(pointID));
        tmpFixedNormals->InsertPoint(pointID, fixedNorms->GetPoint(pointID));
        tmpMovingPoints->InsertPoint(pointID, transformedMovingPoints->GetPoint(pointID));
        tmpMovingNormals->InsertPoint(pointID, transformedMovingNormals->GetPoint(pointID));

        tmpFixedPoints->InsertPoint(pointID2, fixedPoints->GetPoint(pointID2));
        tmpFixedNormals->InsertPoint(pointID2, fixedNorms->GetPoint(pointID2));
        tmpMovingPoints->InsertPoint(pointID2, transformedMovingPoints->GetPoint(pointID2));
        tmpMovingNormals->InsertPoint(pointID2, transformedMovingNormals->GetPoint(pointID2));

        mitk::LiuLeastSquaresWithNormalsRegistrationWrapper::Pointer liuRegistration = mitk::LiuLeastSquaresWithNormalsRegistrationWrapper::New();
        tmpIsSuccessful = liuRegistration->Update(tmpFixedPoints, tmpFixedNormals, tmpMovingPoints, tmpMovingNormals, *tmpRegistrationMatrix, tmpRegistrationError);

        if (tmpIsSuccessful && tmpRegistrationError < bestSoFarRegistrationError)
        {
          bestSoFarPointID1 = pointID;
          bestSoFarPointID2 = pointID2;
          bestSoFarRegistrationError = tmpRegistrationError;
          bestSoFarRegistrationMatrix->DeepCopy(tmpRegistrationMatrix);
        }
      }
    }

    if (bestSoFarRegistrationError < liuFudicialRegistrationError)
    {
      MITK_INFO << "mitk::PointsAndNormalsBasedRegistration: Exhaustive search, FRE=" << bestSoFarRegistrationError << std::endl;
      liuFudicialRegistrationError = bestSoFarRegistrationError;
      liuMatrix->DeepCopy(bestSoFarRegistrationMatrix);
    }
  }

  if (m_UseTwoPhase && liuFudicialRegistrationError > arunFudicialRegistrationError)
  {
    // Fallback to just Arun method, using Normals as fake points.
    outputTransform.DeepCopy(arunMatrix);
    fiducialRegistrationError = arunFudicialRegistrationError;
    MITK_WARN << "mitk::PointsAndNormalsBasedRegistration: Liu's method was used, but falling back to just Arun's method" << std::endl;
  }
  else
  {
    // Combine the two transformations, and report the final error.
    vtkMatrix4x4::Multiply4x4(liuMatrix, arunMatrix, &outputTransform);
    fiducialRegistrationError = liuFudicialRegistrationError;
  }

  return isSuccessful;
}

} // end namespace

