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
const bool mitk::PointBasedRegistration::DEFAULT_USE_POINT_ID_TO_MATCH(false);

namespace mitk
{

//-----------------------------------------------------------------------------
PointBasedRegistration::PointBasedRegistration()
: m_UseICPInitialisation(DEFAULT_USE_ICP_INITIALISATION)
, m_UsePointIDToMatchPoints(DEFAULT_USE_POINT_ID_TO_MATCH)
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

  double error = std::numeric_limits<double>::max();

  mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::ConstPointer transform = NULL;
  mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::MatrixType rotationMatrix;
  mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::OffsetType translationVector;

  rotationMatrix.SetIdentity();
  translationVector.Fill(0);

  if (m_UsePointIDToMatchPoints)
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

      MITK_INFO << "PointBasedRegistration: Matched FRE=" << matchedRegistration->GetFRE() << "mm (Std. Dev. " << matchedRegistration->GetFREStdDev() << ")" << std::endl;
      MITK_INFO << "PointBasedRegistration: Matched RMS=" << matchedRegistration->GetRMSError() << "mm " << std::endl;
      MITK_INFO << "PointBasedRegistration: Matched min=" << matchedRegistration->GetMinError() << "mm" << std::endl;
      MITK_INFO << "PointBasedRegistration: Matched max=" << matchedRegistration->GetMaxError() << "mm" << std::endl;

      error = matchedRegistration->GetFRE();
      transform = matchedRegistration->GetLandmarkTransform();
      rotationMatrix = transform->GetMatrix();
      translationVector = transform->GetOffset();
    }
  }
  else
  {
    mitk::NavigationDataLandmarkTransformFilter::Pointer transformFilter = mitk::NavigationDataLandmarkTransformFilter::New();
    transformFilter->SetUseICPInitialization(m_UseICPInitialisation);
    transformFilter->SetTargetLandmarks(fixedPointSet);
    transformFilter->SetSourceLandmarks(movingPointSet);
    transformFilter->Update();

    MITK_INFO << "PointBasedRegistration: FRE=" << transformFilter->GetFRE() << "mm (Std. Dev. " << transformFilter->GetFREStdDev() << ")" << std::endl;
    MITK_INFO << "PointBasedRegistration: RMS=" << transformFilter->GetRMSError() << "mm " << std::endl;
    MITK_INFO << "PointBasedRegistration: min=" << transformFilter->GetMinError() << "mm" << std::endl;
    MITK_INFO << "PointBasedRegistration: max=" << transformFilter->GetMaxError() << "mm" << std::endl;

    error = transformFilter->GetFRE();
    transform = transformFilter->GetLandmarkTransform();
    rotationMatrix = transform->GetMatrix();
    translationVector = transform->GetOffset();
  }

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

