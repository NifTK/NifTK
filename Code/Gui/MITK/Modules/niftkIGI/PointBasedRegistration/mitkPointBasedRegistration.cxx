/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPointBasedRegistration.h"

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
void PointBasedRegistration::Update(
    const mitk::PointSet::Pointer fixedPointSet,
    const mitk::PointSet::Pointer movingPointSet,
    vtkMatrix4x4& outputTransform)
{

}

} // end namespace

