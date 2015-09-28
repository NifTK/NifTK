/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPointSetUpdate.h"
#include "mitkPointUtils.h"

namespace mitk
{

//-----------------------------------------------------------------------------
PointSetUpdate::PointSetUpdate(
    mitk::OperationType type,
    mitk::PointSet::Pointer pointSet
    )
: mitk::Operation(type)
{
  m_PointSet = mitk::PointSet::New();
  if (pointSet.IsNotNull() && pointSet->GetSize() > 0)
  {
    mitk::CopyPointSets(*pointSet, *m_PointSet);
  }
}

//-----------------------------------------------------------------------------
PointSetUpdate::~PointSetUpdate()
{
}


//-----------------------------------------------------------------------------
void PointSetUpdate::AppendPoint(const mitk::Point3D& point)
{
  int currentSize = m_PointSet->GetSize();
  m_PointSet->InsertPoint(currentSize, point);
}

} // end namespace
