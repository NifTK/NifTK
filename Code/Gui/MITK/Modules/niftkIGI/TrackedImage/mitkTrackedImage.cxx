/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackedImage.h"
#include <mitkCoordinateAxesData.h>
#include <mitkMathsUtils.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <mitkImage.h>
#include <mitkSurface.h>

namespace mitk
{

const char* TrackedImage::TRACKED_IMAGE_SELECTED_PROPERTY_NAME("niftk.trackedimage");

//-----------------------------------------------------------------------------
TrackedImage::TrackedImage()
{
}


//-----------------------------------------------------------------------------
TrackedImage::~TrackedImage()
{
}


//-----------------------------------------------------------------------------
void TrackedImage::Update(const mitk::DataNode::Pointer imageNode,
                                 const mitk::DataNode::Pointer trackingSensorToTrackerNode,
                                 const vtkMatrix4x4& imageToTrackingSensor,
                                 const mitk::Point2D& imageScaling
                                 )
{
  if (imageNode.IsNull())
  {
    MITK_ERROR << "TrackedImage::Update, invalid imageNode";
    return;
  }

  mitk::CoordinateAxesData::Pointer trackingSensorToWorld = dynamic_cast<mitk::CoordinateAxesData*>(trackingSensorToTrackerNode->GetData());
  if (trackingSensorToWorld.IsNull())
  {
    MITK_ERROR << "TrackedImage::Update, invalid trackingSensorToWorld";
    return;
  }

  vtkSmartPointer<vtkMatrix4x4> trackingSensorToWorldTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  trackingSensorToWorld->GetVtkMatrix(*trackingSensorToWorldTransform);

  vtkSmartPointer<vtkMatrix4x4> combinedTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  combinedTransform->Identity();

  combinedTransform->Multiply4x4(trackingSensorToWorldTransform, &imageToTrackingSensor, combinedTransform);

  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
  if (image.IsNotNull())
  {
    mitk::BaseGeometry* geometry = image->GetGeometry();
    if (geometry)
    {
      mitk::Vector3D spacing = geometry->GetSpacing();
      if (! (mitk::IsCloseToZero(spacing[0] - imageScaling[0])
          && mitk::IsCloseToZero(spacing[1] - imageScaling[1])))
      {
        // This is surprisingly expensive, so avoid it if possible.
        geometry->SetSpacing(spacing);
      }

      geometry->SetIndexToWorldTransformByVtkMatrix(combinedTransform);
    }
  }
}

} // end namespace

