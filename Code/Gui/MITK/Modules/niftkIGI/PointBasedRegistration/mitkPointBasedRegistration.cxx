/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPointBasedRegistration.h"
#include <mitkFileIOUtils.h>
#include <mitkNavigationDataLandmarkTransformFilter.h>
#include <mitkPointUtils.h>

const bool mitk::PointBasedRegistration::DEFAULT_USE_ICP_INITIALISATION(false);

namespace mitk
{

//-----------------------------------------------------------------------------
PointBasedRegistration::PointBasedRegistration()
: m_AlwaysTryMatchedPoints(false)
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
    const bool& useICPInitialisation,
    vtkMatrix4x4& outputTransform) const
{

  double error = std::numeric_limits<double>::max();
  bool haveDoneRegistration = false;

  mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::ConstPointer transform = NULL;
  mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::MatrixType rotationMatrix;
  mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::OffsetType translationVector;

  // MITK class NavigationDataLandmarkTransformFilter has the following constraints.
  if ((!useICPInitialisation && fixedPointSet->GetSize() >= 3 && movingPointSet->GetSize() >= 3 && fixedPointSet->GetSize() == movingPointSet->GetSize())
    || (useICPInitialisation && fixedPointSet->GetSize() >= 6 && movingPointSet->GetSize() >= 6)
    )
  {
    mitk::NavigationDataLandmarkTransformFilter::Pointer registrationFilter = mitk::NavigationDataLandmarkTransformFilter::New();
    registrationFilter->SetUseICPInitialization(useICPInitialisation);
    registrationFilter->SetTargetLandmarks(fixedPointSet);
    registrationFilter->SetSourceLandmarks(movingPointSet);
    registrationFilter->Update();

    MITK_INFO << "PointBasedRegistration: FRE=" << registrationFilter->GetFRE() << "mm (Std. Dev. " << registrationFilter->GetFREStdDev() << ")" << std::endl;
    MITK_INFO << "PointBasedRegistration: RMS=" << registrationFilter->GetRMSError() << "mm " << std::endl;
    MITK_INFO << "PointBasedRegistration: min=" << registrationFilter->GetMinError() << "mm" << std::endl;
    MITK_INFO << "PointBasedRegistration: max=" << registrationFilter->GetMaxError() << "mm" << std::endl;

    haveDoneRegistration = true;
    error = registrationFilter->GetFRE();

    transform = registrationFilter->GetLandmarkTransform();
    rotationMatrix = transform->GetMatrix();
    translationVector = transform->GetOffset();
  }

  // In the event that we couldn't do anything above, try the following fallback.
  if (!haveDoneRegistration || m_AlwaysTryMatchedPoints)
  {
    mitk::PointSet::Pointer filteredFixedPoints = mitk::PointSet::New();
    mitk::PointSet::Pointer filteredMovingPoints = mitk::PointSet::New();
    int numberOfFilteredPoints = mitk::FilterMatchingPoints(*fixedPointSet,
                                                            *movingPointSet,
                                                            *filteredFixedPoints,
                                                            *filteredMovingPoints
                                                           );
    if (numberOfFilteredPoints >= 3)
    {
      mitk::NavigationDataLandmarkTransformFilter::Pointer matchedRegistration = mitk::NavigationDataLandmarkTransformFilter::New();
      matchedRegistration->SetUseICPInitialization(false);
      matchedRegistration->SetTargetLandmarks(filteredFixedPoints);
      matchedRegistration->SetSourceLandmarks(filteredMovingPoints);
      matchedRegistration->Update();
      error = matchedRegistration->GetFRE();

      MITK_INFO << "PointBasedRegistration: Matched FRE=" << matchedRegistration->GetFRE() << "mm (Std. Dev. " << matchedRegistration->GetFREStdDev() << ")" << std::endl;
      MITK_INFO << "PointBasedRegistration: Matched RMS=" << matchedRegistration->GetRMSError() << "mm " << std::endl;
      MITK_INFO << "PointBasedRegistration: Matched min=" << matchedRegistration->GetMinError() << "mm" << std::endl;
      MITK_INFO << "PointBasedRegistration: Matched max=" << matchedRegistration->GetMaxError() << "mm" << std::endl;

      double tmpError = matchedRegistration->GetFRE();

      if (tmpError < error)
      {
        haveDoneRegistration = true;
        error = matchedRegistration->GetFRE();

        transform = matchedRegistration->GetLandmarkTransform();
        rotationMatrix = transform->GetMatrix();
        translationVector = transform->GetOffset();
      }
    }
  }

  // Output the result.
  outputTransform.Identity();
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      outputTransform.SetElement(i, j, rotationMatrix[i][j]);
    }
    outputTransform.SetElement(i, 3, translationVector[i]);
  }

  return error;
}

} // end namespace

