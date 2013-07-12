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
    vtkMatrix4x4& outputTransform) const
{
  return 0;
}

} // end namespace

