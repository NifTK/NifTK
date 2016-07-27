/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleVideoWidget.h"

#include <vtkCamera.h>
#include <vtkTransform.h>

#include <mitkCameraIntrinsics.h>
#include <mitkCameraIntrinsicsProperty.h>

#include <niftkCoordinateAxesData.h>
#include <niftkDataStorageUtils.h>
#include <niftkUndistortion.h>
#include <niftkVTKFunctions.h>

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
, m_UseOverlay(true)
, m_EyeHandFileName("")
, m_EyeHandMatrix(nullptr)
{
  m_BitmapOverlay = niftk::BitmapOverlay::New();
  m_BitmapOverlay->SetRenderWindow(this->GetRenderWindow()->GetRenderer()->GetRenderWindow());

  m_MatrixDrivenCamera = vtkSmartPointer<vtkOpenGLMatrixDrivenCamera>::New();
  this->GetRenderWindow()->GetRenderer()->GetVtkRenderer()->SetActiveCamera(m_MatrixDrivenCamera);
}


//-----------------------------------------------------------------------------
SingleVideoWidget::~SingleVideoWidget()
{
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::SetEyeHandFileName(const std::string& fileName)
{
  if (!fileName.empty())
  {
    // Note: Currently doesn't do error handling properly.
    // i.e no return code, no exception.
    m_EyeHandMatrix = niftk::LoadMatrix4x4FromFile(fileName);
    m_EyeHandFileName = fileName;
    MITK_INFO << "Loading eye-hand matrix:" << m_EyeHandFileName << std::endl;
  }
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::SetDataStorage(mitk::DataStorage* ds)
{
  Single3DViewWidget::SetDataStorage(ds);
  m_BitmapOverlay->SetDataStorage(ds);
  this->SetUseOverlay(m_UseOverlay);
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::SetImageNode(mitk::DataNode* node)
{
  Single3DViewWidget::SetImageNode(node);
  m_BitmapOverlay->SetNode(node);
  this->SetUseOverlay(m_UseOverlay);
}


//-----------------------------------------------------------------------------
void SingleVideoWidget::SetUseOverlay(const bool& useIt)
{
  if (useIt)
  {
    m_BitmapOverlay->Enable();
  }
  else
  {
    m_BitmapOverlay->Disable();
  }
  m_UseOverlay = useIt;
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
  // By default, assume camera position is at origin, looking down the world +ve z-axis.
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

  // If additionally, the user has selected a transformation matrix, we move camera accordingly.
  // Note, 2 use-cases:
  // (a) User specifies camera to world - just use the matrix as given.
  // (b) User specified eye-hand matrix - multiply by eye-hand then tracking matrix
  //                                    - to construct the camera to world.
  if (m_TransformNode.IsNotNull())
  {
    CoordinateAxesData::Pointer transform =
        dynamic_cast<CoordinateAxesData*>(m_TransformNode->GetData());

    if (transform.IsNotNull())
    {
      vtkSmartPointer<vtkMatrix4x4> suppliedMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
      transform->GetVtkMatrix(*suppliedMatrix);

      vtkSmartPointer<vtkMatrix4x4> cameraToWorld = vtkSmartPointer<vtkMatrix4x4>::New();

      if (m_EyeHandFileName.empty())
      {
        // Use case (a) - supplied transform is camera to world.
        cameraToWorld->DeepCopy(suppliedMatrix);
      }
      else
      {
        // Use case (b) - supplied transform is a tracking transform.
        vtkMatrix4x4::Multiply4x4(suppliedMatrix, m_EyeHandMatrix, cameraToWorld);
      }

      cameraToWorld->MultiplyPoint(origin, origin);
      cameraToWorld->MultiplyPoint(focalPoint, focalPoint);
      cameraToWorld->MultiplyPoint(viewUp, viewUp);
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
