/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackedPointerCommand.h"
#include <mitkCoordinateAxesData.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <mitkTimeSlicedGeometry.h>
#include <mitkSurface.h>

const bool mitk::TrackedPointerCommand::UPDATE_VIEW_COORDINATE_DEFAULT(false);

namespace mitk
{

//-----------------------------------------------------------------------------
TrackedPointerCommand::TrackedPointerCommand()
{
}


//-----------------------------------------------------------------------------
TrackedPointerCommand::~TrackedPointerCommand()
{
}


//-----------------------------------------------------------------------------
void TrackedPointerCommand::Update(
         const vtkMatrix4x4* tipToPointerTransform,
         const mitk::DataNode::Pointer pointerToWorldNode,
         const mitk::DataNode::Pointer surfaceNode,
         mitk::Point3D& tipCoordinate
         )
{
  if (tipToPointerTransform == NULL)
  {
    MITK_ERROR << "TrackedPointerCommand::Update, invalid tipToPointerTransform";
    return;
  }

  mitk::CoordinateAxesData::Pointer pointerToWorld = dynamic_cast<mitk::CoordinateAxesData*>(pointerToWorldNode->GetData());
  if (pointerToWorld.IsNull())
  {
    MITK_ERROR << "TrackedPointerCommand::Update, invalid pointerToWorldNode";
    return;
  }

  mitk::Surface::Pointer surface = dynamic_cast<mitk::Surface*>(surfaceNode->GetData());
  if (surface.IsNull())
  {
    MITK_ERROR << "TrackedPointerCommand::Update, invalid surfaceNode";
    return;
  }

  vtkSmartPointer<vtkMatrix4x4> pointerToWorldTransform = vtkMatrix4x4::New();
  pointerToWorld->GetVtkMatrix(*pointerToWorldTransform);

  vtkSmartPointer<vtkMatrix4x4> combinedTransform = vtkMatrix4x4::New();
  combinedTransform->Identity();

  combinedTransform->Multiply4x4(tipToPointerTransform, pointerToWorldTransform, combinedTransform);

  mitk::Geometry3D::Pointer geometry = surface->GetGeometry();
  if (geometry.IsNotNull())
  {
    geometry->SetIndexToWorldTransformByVtkMatrix(combinedTransform);
    geometry->Modified();
  }

  double coordinateIn[4] = {0, 0, 0, 1};
  double coordinateOut[4] = {0, 0, 0, 1};

  coordinateIn[0] = tipCoordinate[0];
  coordinateIn[1] = tipCoordinate[1];
  coordinateIn[2] = tipCoordinate[2];

  combinedTransform->MultiplyPoint(coordinateIn, coordinateOut);

  tipCoordinate[0] = coordinateOut[0];
  tipCoordinate[1] = coordinateOut[1];
  tipCoordinate[2] = coordinateOut[2];
}

//-----------------------------------------------------------------------------
} // end namespace

