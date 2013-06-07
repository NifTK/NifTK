/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackedImageCommand.h"
#include <mitkCoordinateAxesData.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <mitkTimeSlicedGeometry.h>
#include <mitkSurface.h>

namespace mitk
{

//-----------------------------------------------------------------------------
TrackedImageCommand::TrackedImageCommand()
{
}


//-----------------------------------------------------------------------------
TrackedImageCommand::~TrackedImageCommand()
{
}


//-----------------------------------------------------------------------------
void TrackedImageCommand::Update(const mitk::DataNode::Pointer imageNode,
                                 const mitk::DataNode::Pointer surfaceNode,
                                 const mitk::DataNode::Pointer probeToWorldNode,
                                 const vtkMatrix4x4* imageToProbeTransform)
{
  if (imageToProbeTransform == NULL)
  {
    MITK_ERROR << "TrackedImageCommand::Update, invalid imageToProbeTransform";
    return;
  }

  mitk::CoordinateAxesData::Pointer probeToWorld = dynamic_cast<mitk::CoordinateAxesData*>(probeToWorldNode->GetData());
  if (probeToWorld.IsNull())
  {
    MITK_ERROR << "TrackedImageCommand::Update, invalid probeToWorldNode";
    return;
  }

  vtkSmartPointer<vtkMatrix4x4> probeToWorldTransform = vtkMatrix4x4::New();
  probeToWorld->GetVtkMatrix(*probeToWorldTransform);

  vtkSmartPointer<vtkMatrix4x4> combinedTransform = vtkMatrix4x4::New();
  combinedTransform->Identity();

  combinedTransform->Multiply4x4(probeToWorldTransform, imageToProbeTransform, combinedTransform);

  mitk::Surface::Pointer surface = dynamic_cast<mitk::Surface*>(surfaceNode->GetData());
  if (surface.IsNotNull())
  {
    mitk::Geometry3D::Pointer geometry = surface->GetGeometry();
    if (geometry.IsNotNull())
    {
      geometry->SetIndexToWorldTransformByVtkMatrix(combinedTransform);
      geometry->Modified();
    }
  }

  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
  if (image.IsNotNull())
  {
    mitk::Geometry3D::Pointer geometry = image->GetGeometry();
    if (geometry.IsNotNull())
    {
      geometry->SetIndexToWorldTransformByVtkMatrix(combinedTransform);
      geometry->Modified();
      imageNode->Modified();
    }
  }
}

} // end namespace

