/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackedImage.h"

#include <vtkImageData.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

#include <mitkImage.h>
#include <mitkSurface.h>

#include <niftkCoordinateAxesData.h>

namespace mitk
{

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
                                 const vtkMatrix4x4& emToOptical
                                 )
{
  if (imageNode.IsNull())
  {
    MITK_ERROR << "TrackedImage::Update, invalid imageNode";
    return;
  }

  niftk::CoordinateAxesData::Pointer trackingSensorToWorld = dynamic_cast<niftk::CoordinateAxesData*>(trackingSensorToTrackerNode->GetData());
  if (trackingSensorToWorld.IsNull())
  {
    MITK_ERROR << "TrackedImage::Update, invalid trackingSensorToWorld";
    return;
  }

  vtkSmartPointer<vtkMatrix4x4> trackingSensorToWorldTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  trackingSensorToWorld->GetVtkMatrix(*trackingSensorToWorldTransform);

  vtkSmartPointer<vtkMatrix4x4> combinedTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  combinedTransform->Identity();
  vtkMatrix4x4::Multiply4x4(trackingSensorToWorldTransform, &imageToTrackingSensor, combinedTransform);

  vtkSmartPointer<vtkMatrix4x4> image2world = vtkSmartPointer<vtkMatrix4x4>::New();
  image2world->Identity();
  vtkMatrix4x4::Multiply4x4(&emToOptical, combinedTransform, image2world);

  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
  if (image.IsNotNull())
  {
    mitk::BaseGeometry* geometry = image->GetGeometry();
    if (geometry)
    {
      geometry->SetIndexToWorldTransformByVtkMatrix(image2world);
      geometry->Modified();
    }
    image->GetVtkImageData()->Modified();
  }
}

} // end namespace

