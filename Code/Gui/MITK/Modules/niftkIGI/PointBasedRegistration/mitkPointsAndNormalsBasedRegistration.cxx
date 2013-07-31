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
double PointsAndNormalsBasedRegistration::Update(
    const mitk::PointSet::Pointer fixedPointSet,
    const mitk::PointSet::Pointer movingPointSet,
    const mitk::PointSet::Pointer fixedNormals,
    const mitk::PointSet::Pointer movingNormals,
    vtkMatrix4x4& outputTransform) const
{
  assert(fixedPointSet);
  assert(movingPointSet);
  assert(fixedNormals);
  assert(movingNormals);

  double fiducialRegistrationError = std::numeric_limits<double>::max();
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
      MITK_ERROR << "mitk::PointsAndNormalsBasedRegistration: filteredFixedPoints size=" << filteredFixedPoints->GetSize() << ", filteredMovingPoints size=" << filteredMovingPoints->GetSize() << ", abandoning use of filtered data sets.";
      return fiducialRegistrationError;
    }

    if (numberOfFilteredNormals >= 2)
    {
      fixedNorms = filteredFixedNormals;
      movingNorms = filteredMovingNormals;
    }
    else
    {
      MITK_ERROR << "mitk::PointsAndNormalsBasedRegistration: filteredFixedNormals size=" << filteredFixedNormals->GetSize() << ", filteredMovingNormals size=" << filteredMovingNormals->GetSize() << ", abandoning use of filtered data sets.";
      return fiducialRegistrationError;
    }

    if (numberOfFilteredPoints != numberOfFilteredNormals)
    {
      MITK_ERROR << "mitk::PointsAndNormalsBasedRegistration: numberOfFilteredPoints=" << numberOfFilteredPoints << ", numberOfFilteredNormals=" << numberOfFilteredNormals << ", abandoning use of filtered data sets.";
      return fiducialRegistrationError;
    }
  }

  if (fixedPoints->GetSize() < 2 || movingPoints->GetSize() < 2)
  {
    MITK_ERROR << "mitk::PointsAndNormalsBasedRegistration:: fixedPoints size=" << fixedPoints->GetSize() << ", movingPoints size=" << movingPoints->GetSize() << ", abandoning point based registration";
    return fiducialRegistrationError;
  }

  mitk::LiuLeastSquaresWithNormalsRegistrationWrapper::Pointer registration = mitk::LiuLeastSquaresWithNormalsRegistrationWrapper::New();
  bool success = registration->Update(fixedPointSet, fixedNorms, movingPointSet, movingNorms, outputTransform, fiducialRegistrationError);

  if (!success)
  {
    MITK_ERROR << "mitk::PointsAndNormalsBasedRegistration: SVD method failed" << std::endl;
  }

  return fiducialRegistrationError;
}

} // end namespace

