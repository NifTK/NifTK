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

const std::string TrackedImageCommand::TRACKED_IMAGE_NODE_NAME("TrackedImageViewPlane");

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
                                 const mitk::DataNode::Pointer trackingSensorToTrackerNode,
                                 const vtkMatrix4x4* imageToTrackingSensor,
                                 const mitk::Point2D& imageScaling
                                 )
{
  if (imageNode.IsNull())
  {
    MITK_ERROR << "TrackedImageCommand::Update, invalid imageNode";
    return;
  }

  mitk::CoordinateAxesData::Pointer trackingSensorToWorld = dynamic_cast<mitk::CoordinateAxesData*>(trackingSensorToTrackerNode->GetData());
  if (trackingSensorToWorld.IsNull())
  {
    MITK_ERROR << "TrackedImageCommand::Update, invalid trackingSensorToWorld";
    return;
  }

  if (imageToTrackingSensor == NULL)
  {
    MITK_ERROR << "TrackedImageCommand::Update, invalid imageToTrackingSensor";
    return;
  }

  vtkSmartPointer<vtkMatrix4x4> trackingSensorToWorldTransform = vtkMatrix4x4::New();
  trackingSensorToWorld->GetVtkMatrix(*trackingSensorToWorldTransform);

  vtkSmartPointer<vtkMatrix4x4> combinedTransform = vtkMatrix4x4::New();
  combinedTransform->Identity();

  combinedTransform->Multiply4x4(trackingSensorToWorldTransform, imageToTrackingSensor, combinedTransform);

  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
  if (image.IsNotNull())
  {
    mitk::Geometry3D::Pointer geometry = image->GetGeometry();
    if (geometry.IsNotNull())
    {
      mitk::Vector3D spacing = geometry->GetSpacing();
      spacing[0] = imageScaling[0];
      spacing[1] = imageScaling[1];

      geometry->SetIndexToWorldTransformByVtkMatrix(combinedTransform);
      geometry->SetSpacing(spacing);
    }
  }
}

} // end namespace

