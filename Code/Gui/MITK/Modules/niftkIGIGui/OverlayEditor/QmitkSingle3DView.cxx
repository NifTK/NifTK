/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkSingle3DView.h"
#include <QGridLayout>
#include <mitkDataStorageUtils.h>
#include <mitkCoordinateAxesData.h>
#include <vtkCamera.h>
#include <vtkTransform.h>
#include <mitkCameraIntrinsics.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <Undistortion.h>

//-----------------------------------------------------------------------------
QmitkSingle3DView::QmitkSingle3DView(QWidget* parent, Qt::WindowFlags f, mitk::RenderingManager* renderingManager)
: QWidget(parent, f)
, m_DataStorage(NULL)
, m_RenderWindow(NULL)
, m_Layout(NULL)
, m_RenderingManager(NULL)
, m_RenderWindowFrame(NULL)
, m_GradientBackground(NULL)
, m_LogoRendering(NULL)
, m_BitmapOverlay(NULL)
, m_TrackingCalibrationFileName("")
, m_TrackingCalibrationTransform(NULL)
, m_TransformNode(NULL)
, m_MatrixDrivenCamera(NULL)
{
  /******************************************************
   * Use the global RenderingManager if none was specified
   ******************************************************/
  if (m_RenderingManager == NULL)
  {
    m_RenderingManager = mitk::RenderingManager::GetInstance();
  }

  m_RenderWindow = new QmitkRenderWindow(this, "single.widget1", NULL, m_RenderingManager);
  m_RenderWindow->setMaximumSize(2000,2000);
  m_RenderWindow->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  m_Layout = new QGridLayout(this);
  m_Layout->setContentsMargins(0, 0, 0, 0);
  m_Layout->addWidget(m_RenderWindow);

  mitk::BaseRenderer::GetInstance(m_RenderWindow->GetRenderWindow())->SetMapperID(mitk::BaseRenderer::Standard3D);

  m_BitmapOverlay = QmitkBitmapOverlay::New();
  m_BitmapOverlay->SetRenderWindow(m_RenderWindow->GetRenderWindow());

  m_RenderWindow->GetRenderer()->GetVtkRenderer()->InteractiveOff();
  m_RenderWindow->GetVtkRenderWindow()->GetInteractor()->Disable();

  m_GradientBackground = mitk::GradientBackground::New();
  m_GradientBackground->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_GradientBackground->SetGradientColors(0, 0, 0, 0, 0, 0);
  m_GradientBackground->Enable();

  m_LogoRendering = CMICLogo::New();
  m_LogoRendering->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_LogoRendering->Disable();

  m_RenderWindowFrame = mitk::RenderWindowFrame::New();
  m_RenderWindowFrame->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_RenderWindowFrame->Enable(1.0,0.0,0.0);

  m_TrackingCalibrationTransform = vtkMatrix4x4::New();
  m_TrackingCalibrationTransform->Identity();

  m_MatrixDrivenCamera = vtkOpenGLMatrixDrivenCamera::New();
  this->GetRenderWindow()->GetRenderer()->GetVtkRenderer()->SetActiveCamera(m_MatrixDrivenCamera);

}


//-----------------------------------------------------------------------------
QmitkSingle3DView::~QmitkSingle3DView()
{
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetDataStorage( mitk::DataStorage* ds )
{
  m_DataStorage = ds;
  mitk::BaseRenderer::GetInstance(m_RenderWindow->GetRenderWindow())->SetDataStorage(ds);

  if (m_DataStorage.IsNotNull())
  {
    m_BitmapOverlay->SetDataStorage (m_DataStorage);
    m_BitmapOverlay->Enable();
  }
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkSingle3DView::GetRenderWindow() const
{
  return m_RenderWindow;
}


//-----------------------------------------------------------------------------
float QmitkSingle3DView::GetOpacity() const
{
  return static_cast<float>(m_BitmapOverlay->GetOpacity());
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetOpacity(const float& value)
{
  m_BitmapOverlay->SetOpacity(value);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetImageNode(const mitk::DataNode* node)
{
  m_BitmapOverlay->SetNode(node);

  bool useDefaultVTKCameraBehaviour = true;

  if (node != NULL)
  {
    mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
    if (image != NULL)
    {
      int width = image->GetDimension(0);
      int height = image->GetDimension(1);
      m_MatrixDrivenCamera->SetCalibratedImageSize(width, height);

      // Check for property that determines if we are doing a calibrated model or not.
      mitk::CameraIntrinsicsProperty::Pointer intrinsicsProperty
          = dynamic_cast<mitk::CameraIntrinsicsProperty*>(node->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName));

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

        useDefaultVTKCameraBehaviour = false;
      }
    }
  }

  m_MatrixDrivenCamera->SetDefaultBehaviour(useDefaultVTKCameraBehaviour);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetTransformNode(const mitk::DataNode* node)
{
  if (node != NULL)
  {
    m_TransformNode = const_cast<mitk::DataNode*>(node);
  }
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::EnableGradientBackground()
{
  m_GradientBackground->Enable();
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::DisableGradientBackground()
{
  m_GradientBackground->Disable();
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetGradientBackgroundColors( const mitk::Color & upper, const mitk::Color & lower )
{
  m_GradientBackground->SetGradientColors(upper[0], upper[1], upper[2], lower[0], lower[1], lower[2]);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::EnableDepartmentLogo()
{
   m_LogoRendering->Enable();
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::DisableDepartmentLogo()
{
   m_LogoRendering->Disable();
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetDepartmentLogoPath( const char * path )
{
  m_LogoRendering->SetLogoSource(path);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::resizeEvent(QResizeEvent* /*event*/)
{
  if (this->isVisible())
  {
    m_BitmapOverlay->SetupCamera();
    this->Update();
  }
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetTrackingCalibrationFileName(const std::string& fileName)
{
  if (m_DataStorage.IsNotNull() && fileName.size() > 0 && fileName != this->m_TrackingCalibrationFileName)
  {
    LoadMatrixOrCreateDefault(fileName, "niftk.ov.cal", true /* helper object */, m_DataStorage);
    m_TrackingCalibrationFileName = fileName;
  }
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::Update()
{
  double znear = 0.01;
  double zfar = 1001;

  int widthOfCurrentWindow = this->width();
  int heightOfCurrentWindow = this->height();

  // So we set the window size on each update so that the OpenGL viewport is always up to date.
  m_MatrixDrivenCamera->SetActualWindowSize(widthOfCurrentWindow, heightOfCurrentWindow);

  // This implies a right handed coordinate system.
  double origin[4]     = {0, 0,     0,    1};
  double focalPoint[4] = {0, 0,     2000, 1};
  double viewUp[4]     = {0, -1.0e9, 0,    1};

  // By default, looking down the world z-axis.
  m_MatrixDrivenCamera->SetPosition(origin[0], origin[1], origin[2]);
  m_MatrixDrivenCamera->SetFocalPoint(focalPoint[0], focalPoint[1], focalPoint[2]);
  m_MatrixDrivenCamera->SetViewUp(viewUp[0], viewUp[1], viewUp[2]);
  m_MatrixDrivenCamera->SetClippingRange(znear, zfar);

  // If we have a calibration and tracking matrix, we can move camera accordingly.
  if (m_TransformNode.IsNotNull() && m_TrackingCalibrationTransform != NULL)
  {
    mitk::CoordinateAxesData::Pointer trackingTransform = dynamic_cast<mitk::CoordinateAxesData*>(m_TransformNode->GetData());
    if (trackingTransform.IsNotNull())
    {
      // And now do the extrinsics. We basically need to move the camera to the right
      // position, and set focal point and view up in world (tracker coordinates).

      // This is achieved by taking the default camera position, focal point and
      // view up specified above and multiply by the calibration (eye to hand transform)
      // matrix, and then by the tracker matrix, which transforms from the hand to
      // tracker coordinates. If the "calibration" is a hand-eye rather than an eye-hand
      // then the calibration matrix may well need inverting. I haven't been able to test this
      // as the moment, as the test machine is broken.

      vtkSmartPointer<vtkMatrix4x4> trackingTransformMatrix = vtkMatrix4x4::New();
      trackingTransform->GetVtkMatrix(*trackingTransformMatrix);

      vtkSmartPointer<vtkMatrix4x4> combinedTransform = vtkMatrix4x4::New();
      vtkMatrix4x4::Multiply4x4(trackingTransformMatrix, m_TrackingCalibrationTransform, combinedTransform);

      double transformedOrigin[4]     = {0, 0, 0, 1};
      double transformedFocalPoint[4] = {0, 0, 0, 1};
      double transformedViewUp[4]     = {0, 0, 0, 1};

      combinedTransform->MultiplyPoint(origin, transformedOrigin);
      combinedTransform->MultiplyPoint(focalPoint, transformedFocalPoint);
      combinedTransform->MultiplyPoint(viewUp, transformedViewUp);

      m_MatrixDrivenCamera->SetPosition(transformedOrigin[0], transformedOrigin[1], transformedOrigin[2]);
      m_MatrixDrivenCamera->SetFocalPoint(transformedFocalPoint[0], transformedFocalPoint[1], transformedFocalPoint[2]);
      m_MatrixDrivenCamera->SetViewUp(transformedViewUp[0], transformedViewUp[1], transformedViewUp[2]);
      m_MatrixDrivenCamera->SetClippingRange(znear, zfar);
    }
  }
}



