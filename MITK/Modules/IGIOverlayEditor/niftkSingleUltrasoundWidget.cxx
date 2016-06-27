/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleUltrasoundWidget.h"
#include <niftkVTKFunctions.h>
#include <mitkImage2DToTexturePlaneMapper3D.h>

namespace niftk
{
//-----------------------------------------------------------------------------
SingleUltrasoundWidget::SingleUltrasoundWidget(QWidget* parent,
                                               Qt::WindowFlags f,
                                               mitk::RenderingManager* renderingManager)
: Single3DViewWidget(parent, f, renderingManager)
, m_ClipToImagePlane(true)
{
}


//-----------------------------------------------------------------------------
SingleUltrasoundWidget::~SingleUltrasoundWidget()
{
  this->RemoveTextureMapper();
}


//-----------------------------------------------------------------------------
void SingleUltrasoundWidget::RemoveTextureMapper()
{
  mitk::DataNode::Pointer imageNode = this->m_ImageNode;
  if (imageNode.IsNotNull())
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
    if (image.IsNotNull())
    {
      mitk::Mapper::Pointer mapper = imageNode->GetMapper(mitk::BaseRenderer::Standard3D);
      if (dynamic_cast<mitk::Image2DToTexturePlaneMapper3D*>(mapper.GetPointer()) != NULL)
      {
        imageNode->SetMapper(mitk::BaseRenderer::Standard3D, NULL);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void SingleUltrasoundWidget::SetImageNode(mitk::DataNode* node)
{
  if (node == nullptr
      || (node != nullptr
          && this->m_ImageNode.IsNotNull()
          && dynamic_cast<mitk::Image2DToTexturePlaneMapper3D*>(
             this->m_ImageNode->GetMapper(mitk::BaseRenderer::Standard3D)) != nullptr
         )
      )
  {
    this->RemoveTextureMapper();
  }

  Single3DViewWidget::SetImageNode(node);

  if (node != nullptr)
  {
    mitk::Image2DToTexturePlaneMapper3D::Pointer newMapper = mitk::Image2DToTexturePlaneMapper3D::New();
    node->SetMapper(mitk::BaseRenderer::Standard3D, newMapper);
  }
}


//-----------------------------------------------------------------------------
void SingleUltrasoundWidget::SetClipToImagePlane(const bool& clipToImagePlane)
{
  m_ClipToImagePlane = clipToImagePlane;
}


//-----------------------------------------------------------------------------
void SingleUltrasoundWidget::Update()
{
  // Early exit if the widget itself is not yet on-screen.
  if (!this->isVisible())
  {
    return;
  }
  this->UpdateCameraToTrackImage();
}


//-----------------------------------------------------------------------------
void SingleUltrasoundWidget::UpdateCameraToTrackImage()
{
  if (m_Image.IsNotNull())
  {
    int windowSize[2];
    windowSize[0] = this->width();
    windowSize[1] = this->height();

    int imageSize[2];
    imageSize[0] = m_Image->GetDimension(0);
    imageSize[1] = m_Image->GetDimension(1);

    double distanceToFocalPoint = -1000;
    double clippingRange[2];

    if (m_ClipToImagePlane)
    {
      clippingRange[0] = 999;
      clippingRange[1] = 1001;
    }
    else
    {
      clippingRange[0] = m_ClippingRange[0];
      clippingRange[1] = m_ClippingRange[1];
    }

    double origin[3];
    double spacing[3];
    double xAxis[3];
    double yAxis[3];

    mitk::BaseGeometry* geometry = m_Image->GetGeometry();
    mitk::Point3D geometryOrigin = geometry->GetOrigin();
    mitk::Vector3D geometrySpacing = geometry->GetSpacing();
    mitk::Vector3D geometryXAxis = geometry->GetAxisVector(0);
    mitk::Vector3D geometryYAxis = geometry->GetAxisVector(1);

    for (int i = 0; i < 3; ++i)
    {
      origin[i] = geometryOrigin[i];
      spacing[i] = geometrySpacing[i];
      xAxis[i] = geometryXAxis[i];
      yAxis[i] = geometryYAxis[i];
    }

    vtkCamera *camera = this->GetRenderWindow()->GetRenderer()->GetVtkRenderer()->GetActiveCamera();

    niftk::SetCameraParallelTo2DImage(imageSize, windowSize, origin, spacing,
                                      xAxis, yAxis, clippingRange, true,
                                      *camera,
                                      distanceToFocalPoint);
  }
}

} // end namespace
