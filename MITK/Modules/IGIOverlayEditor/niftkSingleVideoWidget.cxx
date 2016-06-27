/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleVideoWidget.h"
#include <niftkVTKFunctions.h>
#include <niftkUndistortion.h>
#include <mitkCoordinateAxesData.h>
#include <mitkCameraIntrinsics.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkDataStorageUtils.h>

#include <vtkCamera.h>
#include <vtkTransform.h>

namespace niftk
{

//-----------------------------------------------------------------------------
SingleVideoWidget::SingleVideoWidget(QWidget* parent,
                                     Qt::WindowFlags f,
                                     mitk::RenderingManager* renderingManager)
: Single3DViewWidget(parent, f, renderingManager)
, m_BitmapOverlay(nullptr)
, m_TransformNode(nullptr)
, m_MatrixDrivenCamera(nullptr)
, m_IsCalibrated(false)
{
  m_BitmapOverlay = niftk::BitmapOverlay::New();
  m_BitmapOverlay->SetRenderWindow(this->GetRenderWindow()->GetRenderer()->GetRenderWindow());
  m_BitmapOverlay->Enable();

  m_MatrixDrivenCamera = vtkSmartPointer<vtkOpenGLMatrixDrivenCamera>::New();
  this->GetRenderWindow()->GetRenderer()->GetVtkRenderer()->SetActiveCamera(m_MatrixDrivenCamera);
}


//-----------------------------------------------------------------------------
SingleVideoWidget::~SingleVideoWidget()
{
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::SetDataStorage(mitk::DataStorage* ds)
{
  m_BitmapOverlay->Disable();
  Single3DViewWidget::SetDataStorage(ds);
  m_BitmapOverlay->SetDataStorage(ds);
  m_BitmapOverlay->Enable();
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::SetImageNode(mitk::DataNode* node)
{
  m_BitmapOverlay->Disable();
  Single3DViewWidget::SetImageNode(node);
  m_BitmapOverlay->SetNode(node);
  m_BitmapOverlay->Enable();
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::NodeRemoved(const mitk::DataNode* node)
{
  m_BitmapOverlay->NodeRemoved(node);
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::NodeChanged(const mitk::DataNode* node)
{
  m_BitmapOverlay->NodeChanged(node);
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::NodeAdded(const mitk::DataNode* node)
{
  m_BitmapOverlay->NodeAdded(node);
}

//-----------------------------------------------------------------------------
void SingleVideoWidget::SetTransformNode(const mitk::DataNode* node)
{
  if (node != NULL)
  {
    m_TransformNode = const_cast<mitk::DataNode*>(node);
  }
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::resizeEvent(QResizeEvent* /*event*/)
{
  m_BitmapOverlay->SetupCamera();
  this->Update();
}


//-----------------------------------------------------------------------------
float SingleVideoWidget::GetOpacity() const
{
  return static_cast<float>(m_BitmapOverlay->GetOpacity());
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::SetOpacity(const float& value)
{
  m_BitmapOverlay->SetOpacity(value);
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::Update()
{
  // Early exit if the widget itself is not yet on-screen.
  if (!this->isVisible())
  {
    return;
  }

  int widthOfCurrentWindow = this->width();
  int heightOfCurrentWindow = this->height();

  this->UpdateCameraIntrinsicParameters();
  m_MatrixDrivenCamera->SetUseCalibratedCamera(m_IsCalibrated);
  m_MatrixDrivenCamera->SetActualWindowSize(widthOfCurrentWindow, heightOfCurrentWindow);
  this->UpdateCameraViaTrackingTransformation();
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::UpdateCameraIntrinsicParameters()
{
  bool isCalibrated = false;

  if (m_Image.IsNotNull() && m_ImageNode.IsNotNull())
  {
    mitk::Vector3D  imgScaling = m_Image->GetGeometry()->GetSpacing();
    int width  = m_Image->GetDimension(0);
    int height = m_Image->GetDimension(1);
    m_MatrixDrivenCamera->SetCalibratedImageSize(width, height, imgScaling[0] / imgScaling[1]);

    // Check for property that determines if we are doing a calibrated model or not.
    mitk::CameraIntrinsicsProperty::Pointer intrinsicsProperty
      = dynamic_cast<mitk::CameraIntrinsicsProperty*>(
        m_ImageNode->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName));

    if (intrinsicsProperty.IsNotNull())
    {
      mitk::CameraIntrinsics::Pointer intrinsics;
      intrinsics = intrinsicsProperty->GetValue();

      m_MatrixDrivenCamera->SetIntrinsicParameters
          (intrinsics->GetFocalLengthX(),
           intrinsics->GetFocalLengthY(),
           intrinsics->GetPrincipalPointX(),
           intrinsics->GetPrincipalPointY()
          );
      isCalibrated = true;
    }
  }
  m_IsCalibrated = isCalibrated;
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::UpdateCameraViaTrackingTransformation()
{
  // This implies a right handed coordinate system.
  // By default, assume camera position is at origin, looking down the world z-axis.
  double origin[4]     = {0, 0,    0,    1};
  double focalPoint[4] = {0, 0,   1000, 1};
  double viewUp[4]     = {0, -1000, 0,    1};

  // If the stereo right to left matrix exists, we must be doing the right hand image.
  // So, in this case, we have an extra transformation to consider.
  if (m_Image.IsNotNull())
  {
    niftk::Undistortion::MatrixProperty::Pointer prop
      = dynamic_cast<niftk::Undistortion::MatrixProperty*>(
        m_Image->GetProperty(niftk::Undistortion::s_StereoRigTransformationPropertyName).GetPointer());

    if (prop.IsNotNull())
    {
      itk::Matrix<float, 4, 4> txf = prop->GetValue();
      vtkSmartPointer<vtkMatrix4x4> tmpMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
      for (int i = 0; i < 4; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          tmpMatrix->SetElement(i, j, txf[i][j]);
        }
      }
      tmpMatrix->MultiplyPoint(origin, origin);
      tmpMatrix->MultiplyPoint(focalPoint, focalPoint);
      tmpMatrix->MultiplyPoint(viewUp, viewUp);
      viewUp[0] = viewUp[0] - origin[0];
      viewUp[1] = viewUp[1] - origin[1];
      viewUp[2] = viewUp[2] - origin[2];
    }
  }

  // If additionally, the user has selected a tracking matrix, we can move camera accordingly.
  if (m_TransformNode.IsNotNull())
  {
    mitk::CoordinateAxesData::Pointer trackingTransform = dynamic_cast<mitk::CoordinateAxesData*>(m_TransformNode->GetData());
    if (trackingTransform.IsNotNull())
    {
      vtkSmartPointer<vtkMatrix4x4> trackingTransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
      trackingTransform->GetVtkMatrix(*trackingTransformMatrix);

      trackingTransformMatrix->MultiplyPoint(origin, origin);
      trackingTransformMatrix->MultiplyPoint(focalPoint, focalPoint);
      trackingTransformMatrix->MultiplyPoint(viewUp, viewUp);
      viewUp[0] = viewUp[0] - origin[0];
      viewUp[1] = viewUp[1] - origin[1];
      viewUp[2] = viewUp[2] - origin[2];
    }
  }

  // We then move the camera to that position.
  m_MatrixDrivenCamera->SetPosition(origin[0], origin[1], origin[2]);
  m_MatrixDrivenCamera->SetFocalPoint(focalPoint[0], focalPoint[1], focalPoint[2]);
  m_MatrixDrivenCamera->SetViewUp(viewUp[0], viewUp[1], viewUp[2]);
  m_MatrixDrivenCamera->SetClippingRange(m_ClippingRange[0], m_ClippingRange[1]);
}

} // end namespace


