/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPointsAndNormalsBasedRegistration.h"
#include <mitkFileIOUtils.h>
#include <mitkNavigationDataLandmarkTransformFilter.h>

namespace mitk
{

//-----------------------------------------------------------------------------
PointsAndNormalsBasedRegistration::PointsAndNormalsBasedRegistration()
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

  outputTransform.Identity();


  return 0;
}

} // end namespace

