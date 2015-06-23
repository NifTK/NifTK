/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPointBasedRegistration.h"
#include <limits>
#include <mitkExceptionMacro.h>
#include <mitkPointUtils.h>
#include <mitkNavigationDataLandmarkTransformFilter.h>
#include "niftkArunLeastSquaresPointRegistration.h"

namespace niftk
{

//-----------------------------------------------------------------------------
PointBasedRegistration::PointBasedRegistration()
: m_UseICPInitialisation(PointBasedRegistrationConstants::DEFAULT_USE_ICP_INITIALISATION)
, m_UsePointIDToMatchPoints(PointBasedRegistrationConstants::DEFAULT_USE_POINT_ID_TO_MATCH)
, m_StripNaNFromInput(PointBasedRegistrationConstants::DEFAULT_STRIP_NAN_FROM_INPUT)
{
}


//-----------------------------------------------------------------------------
PointBasedRegistration::~PointBasedRegistration()
{
}


//-----------------------------------------------------------------------------
double PointBasedRegistration::Update(
    const mitk::PointSet::Pointer fixedPointSet,
    const mitk::PointSet::Pointer movingPointSet,
    vtkMatrix4x4& outputTransform) const

{
  if (fixedPointSet.IsNull())
    mitkThrow() << "The 'fixed' points are NULL";
  if (movingPointSet.IsNull())
    mitkThrow() << "The 'moving' points are NULL";

  double fiducialRegistrationError = std::numeric_limits<double>::max();
  outputTransform.Identity();

  mitk::PointSet::Pointer noNaNFixedPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer noNaNMovingPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer filteredFixedPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer filteredMovingPoints = mitk::PointSet::New();
  mitk::PointSet* fixedPoints = fixedPointSet;
  mitk::PointSet* movingPoints = movingPointSet;

  if (m_StripNaNFromInput)
  {
    int fixedPointsRemoved = mitk::RemoveNaNPoints(*fixedPointSet, *noNaNFixedPoints);
    int movingPointsRemoved = mitk::RemoveNaNPoints(*movingPointSet, *noNaNMovingPoints);
    if ( fixedPointsRemoved != 0 )
    {
      MITK_INFO << "Removed " << fixedPointsRemoved << " NaN points from fixed data";
    }
    if ( movingPointsRemoved != 0 )
    {
      MITK_INFO << "Removed " << movingPointsRemoved << " NaN points from moving data";
    }
    fixedPoints = noNaNFixedPoints;
    movingPoints = noNaNMovingPoints;
  }

  if (m_UsePointIDToMatchPoints)
  {
    int numberOfFilteredPoints = mitk::FilterMatchingPoints(*fixedPoints,
                                                            *movingPoints,
                                                            *filteredFixedPoints,
                                                            *filteredMovingPoints
                                                            );
    if (numberOfFilteredPoints < 3)
      mitkThrow() << "After filtering by pointID, there were only "
                  << filteredFixedPoints->GetSize() << " 'fixed' points and "
                  << filteredMovingPoints->GetSize() << " 'moving' points, and we need at least 3";

    fixedPoints = filteredFixedPoints;
    movingPoints = filteredMovingPoints;
  }

  if (m_UseICPInitialisation && fixedPoints->GetSize() >= 6 && movingPoints->GetSize() >= 6)
  {
    mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::ConstPointer transform = NULL;
    mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::MatrixType rotationMatrix;
    mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::OffsetType translationVector;
    rotationMatrix.SetIdentity();
    translationVector.Fill(0);

    mitk::NavigationDataLandmarkTransformFilter::Pointer transformFilter
      = mitk::NavigationDataLandmarkTransformFilter::New();
    transformFilter->SetUseICPInitialization(m_UseICPInitialisation);
    transformFilter->SetTargetLandmarks(fixedPoints);
    transformFilter->SetSourceLandmarks(movingPoints);
    transformFilter->Update();

    MITK_INFO << "PointBasedRegistration: FRE=" << transformFilter->GetFRE()
      << "mm (Std. Dev. " << transformFilter->GetFREStdDev() << ")" << std::endl;
    MITK_INFO << "PointBasedRegistration: RMS=" << transformFilter->GetRMSError()
      << "mm " << std::endl;
    MITK_INFO << "PointBasedRegistration: min=" << transformFilter->GetMinError()
      << "mm" << std::endl;
    MITK_INFO << "PointBasedRegistration: max=" << transformFilter->GetMaxError()
      << "mm" << std::endl;

    fiducialRegistrationError = transformFilter->GetFRE();
    transform = transformFilter->GetLandmarkTransform();
    rotationMatrix = transform->GetMatrix();
    translationVector = transform->GetOffset();

    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        outputTransform.SetElement(i, j, rotationMatrix[i][j]);
      }
      outputTransform.SetElement(i, 3, translationVector[i]);
    }
  }
  else
  {
    // Revert back to SVD. So we only use NavigationDataLandmarkTransformFilter if asked.
    // Also, at this point, there must be exactly the same number of corresponding points, in the same order.
    if (fixedPoints->GetSize() < 3)
      mitkThrow() << "Not enough 'fixed' points for SVD";
    if (movingPoints->GetSize() < 3)
      mitkThrow() << "Not enough 'moving' points for SVD";
    fiducialRegistrationError = niftk::PointBasedRegistrationUsingSVD(fixedPoints, movingPoints, outputTransform);
  }
  return fiducialRegistrationError;
}

} // end namespace
