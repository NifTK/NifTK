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

const bool mitk::PointBasedRegistration::DEFAULT_USE_ICP_INITIALISATION(false);

namespace mitk
{

//-----------------------------------------------------------------------------
PointBasedRegistration::PointBasedRegistration()
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

  mitk::NavigationDataLandmarkTransformFilter::Pointer filter = mitk::NavigationDataLandmarkTransformFilter::New();
  filter->SetUseICPInitialization(useICPInitialisation);
  filter->SetTargetLandmarks(fixedPointSet);
  filter->SetSourceLandmarks(movingPointSet);
  filter->Update();

  MITK_INFO << "PointBasedRegistration: FRE=" << filter->GetFRE() << "mm (Std. Dev. " << filter->GetFREStdDev() << ")" << std::endl;
  MITK_INFO << "PointBasedRegistration: RMS=" << filter->GetRMSError() << "mm " << std::endl;
  MITK_INFO << "PointBasedRegistration: min=" << filter->GetMinError() << "mm" << std::endl;
  MITK_INFO << "PointBasedRegistration: max=" << filter->GetMaxError() << "mm" << std::endl;

  mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::ConstPointer transform = filter->GetLandmarkTransform();
  mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::MatrixType rotationMatrix = transform->GetMatrix();
  mitk::NavigationDataLandmarkTransformFilter::LandmarkTransformType::OffsetType translationVector = transform->GetOffset();

  outputTransform.Identity();
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      outputTransform.SetElement(i, j, rotationMatrix[i][j]);
    }
    outputTransform.SetElement(i, 3, translationVector[i]);
  }
  double error = filter->GetFRE();
  return error;
}

} // end namespace

