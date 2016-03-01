/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPointRegServiceUsingSVD.h"
#include <niftkArunLeastSquaresPointRegistration.h>

namespace niftk
{

//-----------------------------------------------------------------------------
PointRegServiceUsingSVD::PointRegServiceUsingSVD()
{

}


//-----------------------------------------------------------------------------
PointRegServiceUsingSVD::~PointRegServiceUsingSVD()
{

}


//-----------------------------------------------------------------------------
double PointRegServiceUsingSVD::Register(const mitk::PointSet::Pointer fixedPoints,
                                         const mitk::PointSet::Pointer movingPoints,
                                         vtkMatrix4x4& matrix) const
{
  return niftk::PointBasedRegistrationUsingSVD(fixedPoints, movingPoints, matrix);
}

} // end namespace
