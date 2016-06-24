/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingle3DViewWidget.h"
#include <niftkVTKFunctions.h>
#include <niftkUndistortion.h>
#include <mitkCoordinateAxesData.h>
#include <mitkCameraIntrinsics.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkDataStorageUtils.h>

#include <QGridLayout>
#include <mitkBaseGeometry.h>
#include <vtkCamera.h>
#include <vtkTransform.h>

namespace niftk
{
//-----------------------------------------------------------------------------
Single3DViewWidget::Single3DViewWidget(QWidget* parent, Qt::WindowFlags f, mitk::RenderingManager* renderingManager)
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
, m_Image(NULL)
, m_ImageNode(NULL)
, m_MatrixDrivenCamera(NULL)
, m_IsCameraTracking(true)
, m_IsCalibrated(false)
, m_ZNear(2.0)
, m_ZFar(5000)
, m_ClipToImagePlane(true)
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

  m_BitmapOverlay = niftk::BitmapOverlayWidget::New();
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

  m_TrackingCalibrationTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  m_TrackingCalibrationTransform->Identity();

  m_MatrixDrivenCamera = vtkSmartPointer<vtkOpenGLMatrixDrivenCamera>::New();
  this->GetRenderWindow()->GetRenderer()->GetVtkRenderer()->SetActiveCamera(m_MatrixDrivenCamera);
}


//-----------------------------------------------------------------------------
Single3DViewWidget::~Single3DViewWidget()
{
  this->DeRegisterDataStorageListeners();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->RemoveNodeEvent.RemoveListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
       (this, &Single3DViewWidget::NodeRemoved ) );

    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
      (this, &Single3DViewWidget::NodeChanged ) );

    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
      (this, &Single3DViewWidget::NodeAdded ) );
  }
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetDataStorage( mitk::DataStorage* dataStorage )
{
  if (m_DataStorage.IsNotNull() && m_DataStorage != dataStorage)
  {
    this->DeRegisterDataStorageListeners();
  }

  m_DataStorage = dataStorage;

  if (m_DataStorage.IsNotNull())
  {
    m_BitmapOverlay->SetDataStorage (m_DataStorage);
    m_BitmapOverlay->Enable();

    m_DataStorage->RemoveNodeEvent.AddListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
       (this, &Single3DViewWidget::NodeRemoved ) );

    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
      (this, &Single3DViewWidget::NodeChanged ) );

    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
      (this, &Single3DViewWidget::NodeAdded ) );
  }

  mitk::BaseRenderer::GetInstance(m_RenderWindow->GetRenderWindow())->SetDataStorage(dataStorage);
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::NodeAdded(const mitk::DataNode* node)
{
  m_BitmapOverlay->NodeAdded(node);
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::NodeRemoved (const mitk::DataNode * node)
{
  m_BitmapOverlay->NodeRemoved(node);

  if (m_ImageNode.IsNotNull() && node == m_ImageNode)
  {
    this->SetImageNode(NULL);
  }
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::NodeChanged(const mitk::DataNode* node)
{
  m_BitmapOverlay->NodeChanged(node);
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* Single3DViewWidget::GetRenderWindow() const
{
  return m_RenderWindow;
}


//-----------------------------------------------------------------------------
float Single3DViewWidget::GetOpacity() const
{
  return static_cast<float>(m_BitmapOverlay->GetOpacity());
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetOpacity(const float& value)
{
  m_BitmapOverlay->SetOpacity(value);
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetTransformNode(const mitk::DataNode* node)
{
  if (node != NULL)
  {
    m_TransformNode = const_cast<mitk::DataNode*>(node);
  }
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::EnableGradientBackground()
{
  m_GradientBackground->Enable();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::DisableGradientBackground()
{
  m_GradientBackground->Disable();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetGradientBackgroundColors( const mitk::Color & upper, const mitk::Color & lower )
{
  m_GradientBackground->SetGradientColors(upper[0], upper[1], upper[2], lower[0], lower[1], lower[2]);
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::EnableDepartmentLogo()
{
   m_LogoRendering->Enable();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::DisableDepartmentLogo()
{
   m_LogoRendering->Disable();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetDepartmentLogoPath(const QString& path)
{
  m_LogoRendering->SetLogoSource(qPrintable(path));
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::resizeEvent(QResizeEvent* /*event*/)
{
  m_BitmapOverlay->SetupCamera();
  this->Update();
}


//-----------------------------------------------------------------------------
QString Single3DViewWidget::GetTrackingCalibrationFileName() const
{
  return m_TrackingCalibrationFileName;
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetTrackingCalibrationFileName(const QString& fileName)
{
  if (m_DataStorage.IsNotNull() && fileName.size() > 0 && fileName != m_TrackingCalibrationFileName)
  {
    mitk::LoadMatrixOrCreateDefault(fileName.toStdString(), "niftk.ov.cal", true /* helper object */, m_DataStorage);
    m_TrackingCalibrationFileName = fileName;

    mitk::DataNode *node = m_DataStorage->GetNamedNode("niftk.ov.cal");
    if (node != NULL)
    {
      mitk::CoordinateAxesData::Pointer data = dynamic_cast<mitk::CoordinateAxesData*>(node->GetData());
      if (data.IsNotNull())
      {
        data->GetVtkMatrix(*m_TrackingCalibrationTransform);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetCameraTrackingMode(const bool& isCameraTracking)
{
  m_IsCameraTracking = isCameraTracking;
  m_BitmapOverlay->SetEnabled(isCameraTracking);
}


//-----------------------------------------------------------------------------
bool Single3DViewWidget::GetCameraTrackingMode() const
{
  return m_IsCameraTracking;
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetClipToImagePlane(const bool& clipToImagePlane)
{
  m_ClipToImagePlane = clipToImagePlane;
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::UpdateCameraIntrinsicParameters()
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
        = dynamic_cast<mitk::CameraIntrinsicsProperty*>(m_ImageNode->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName));

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
void Single3DViewWidget::SetImageNode(const mitk::DataNode* node)
{
  // Remember: node can be NULL, as we have to respond to NodeRemoved events.

  m_BitmapOverlay->SetNode(node);

  if (node == NULL)
  {
    m_Image = NULL;
    m_ImageNode = NULL;
    m_IsCalibrated = false;
  }
  else
  {
    mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
    if (image != NULL)
    {
      m_Image = image;
      m_ImageNode = const_cast<mitk::DataNode*>(node);
    }
  }
  this->Update();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::Update()
{
  // Early exit if the widget itself is not yet on-screen.
  if (!this->isVisible())
  {
    return;
  }

  this->UpdateCameraIntrinsicParameters();
  m_MatrixDrivenCamera->SetUseCalibratedCamera(m_IsCalibrated);

  int widthOfCurrentWindow = this->width();
  int heightOfCurrentWindow = this->height();
  m_MatrixDrivenCamera->SetActualWindowSize(widthOfCurrentWindow, heightOfCurrentWindow);

  if (m_IsCameraTracking)
  {
    this->UpdateCameraViaTrackingTransformation();
  }
  else
  {
    this->UpdateCameraToTrackImage();
  }
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::UpdateCameraViaTrackingTransformation()
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
    niftk::Undistortion::MatrixProperty::Pointer prop = dynamic_cast<niftk::Undistortion::MatrixProperty*>(m_Image->GetProperty(niftk::Undistortion::s_StereoRigTransformationPropertyName).GetPointer());
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
  if (m_TransformNode.IsNotNull() && m_TrackingCalibrationTransform != NULL)
  {
    mitk::CoordinateAxesData::Pointer trackingTransform = dynamic_cast<mitk::CoordinateAxesData*>(m_TransformNode->GetData());
    if (trackingTransform.IsNotNull())
    {
      vtkSmartPointer<vtkMatrix4x4> trackingTransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
      trackingTransform->GetVtkMatrix(*trackingTransformMatrix);

      vtkSmartPointer<vtkMatrix4x4> combinedTransform = vtkSmartPointer<vtkMatrix4x4>::New();
      vtkMatrix4x4::Multiply4x4( trackingTransformMatrix , m_TrackingCalibrationTransform, combinedTransform);

      combinedTransform->MultiplyPoint(origin, origin);
      combinedTransform->MultiplyPoint(focalPoint, focalPoint);
      combinedTransform->MultiplyPoint(viewUp, viewUp);
      viewUp[0] = viewUp[0] - origin[0];
      viewUp[1] = viewUp[1] - origin[1];
      viewUp[2] = viewUp[2] - origin[2];
    }
  }

  // We then move the camera to that position.
  m_MatrixDrivenCamera->SetPosition(origin[0], origin[1], origin[2]);
  m_MatrixDrivenCamera->SetFocalPoint(focalPoint[0], focalPoint[1], focalPoint[2]);
  m_MatrixDrivenCamera->SetViewUp(viewUp[0], viewUp[1], viewUp[2]);
  m_MatrixDrivenCamera->SetClippingRange(m_ZNear, m_ZFar);
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::UpdateCameraToTrackImage()
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
      clippingRange[0] = m_ZNear;
      clippingRange[1] = m_ZFar;
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

    niftk::SetCameraParallelTo2DImage(imageSize, windowSize, origin, spacing, xAxis, yAxis, clippingRange, true, *m_MatrixDrivenCamera, distanceToFocalPoint);
  }
}

} // end namespace


