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
#include <mitkTrackedImageCommand.h>
#include <mitkCameraIntrinsics.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkGeometry3D.h>
#include <vtkCamera.h>
#include <vtkTransform.h>
#include <vtkFunctions.h>
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
, m_Image(NULL)
, m_ImageNode(NULL)
, m_MatrixDrivenCamera(NULL)
, m_IsCameraTracking(true)
, m_IsCalibrated(false)
, m_ZNear(2.0)
, m_ZFar(5000)
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
  this->DeRegisterDataStorageListeners();
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->RemoveNodeEvent.RemoveListener
      (mitk::MessageDelegate1<QmitkSingle3DView, const mitk::DataNode*>
       (this, &QmitkSingle3DView::NodeRemoved ) );

    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<QmitkSingle3DView, const mitk::DataNode*>
      (this, &QmitkSingle3DView::NodeChanged ) );

    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<QmitkSingle3DView, const mitk::DataNode*>
      (this, &QmitkSingle3DView::NodeAdded ) );
  }
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetDataStorage( mitk::DataStorage* dataStorage )
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
      (mitk::MessageDelegate1<QmitkSingle3DView, const mitk::DataNode*>
       (this, &QmitkSingle3DView::NodeRemoved ) );

    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<QmitkSingle3DView, const mitk::DataNode*>
      (this, &QmitkSingle3DView::NodeChanged ) );

    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<QmitkSingle3DView, const mitk::DataNode*>
      (this, &QmitkSingle3DView::NodeAdded ) );
  }

  mitk::BaseRenderer::GetInstance(m_RenderWindow->GetRenderWindow())->SetDataStorage(dataStorage);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::NodeAdded(const mitk::DataNode* node)
{
  m_BitmapOverlay->NodeAdded(node);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::NodeRemoved (const mitk::DataNode * node )
{
  if ( node == m_ImageNode )
  {
    m_BitmapOverlay->NodeRemoved(node);
    this->SetImageNode(NULL);
  }
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::NodeChanged(const mitk::DataNode* node)
{
  if (m_ImageNode.IsNotNull())
  {
    m_BitmapOverlay->NodeChanged(node);
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
  m_BitmapOverlay->SetupCamera();
  this->Update();
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
void QmitkSingle3DView::SetTrackedImageVisibility(const bool& visibility)
{
  if (m_DataStorage.IsNotNull())
  {
    mitk::DataNode* trackedImage = m_DataStorage->GetNamedNode(mitk::TrackedImageCommand::TRACKED_IMAGE_NODE_NAME);
    if (trackedImage != NULL)
    {
      trackedImage->SetVisibility(visibility, m_RenderWindow->GetRenderer());
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetCameraTrackingMode(const bool& isCameraTracking)
{
  m_IsCameraTracking = isCameraTracking;
  m_BitmapOverlay->SetEnabled(isCameraTracking);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::UpdateCameraIntrinsicParameters()
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
void QmitkSingle3DView::SetImageNode(const mitk::DataNode* node)
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
void QmitkSingle3DView::Update()
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
    this->SetTrackedImageVisibility(false);
    this->UpdateCameraViaTrackingTransformation();
  }
  else
  {
    this->SetTrackedImageVisibility(true);
    this->UpdateCameraToTrackImage();
  }
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::UpdateCameraViaTrackingTransformation()
{
  // This implies a right handed coordinate system.
  // By default, assume camera position is at origin, looking down the world z-axis.
  double origin[4]     = {0, 0,    0,    1};
  double focalPoint[4] = {0, 0,   -1000, 1};
  double viewUp[4]     = {0, 1000, 0,    1};

  // If the stereo right to left matrix exists, we must be doing the right hand image.
  // So, in this case, we have an extra transformation to consider.
  if (m_Image.IsNotNull())
  {
    niftk::Undistortion::MatrixProperty::Pointer prop = dynamic_cast<niftk::Undistortion::MatrixProperty*>(m_Image->GetProperty(niftk::Undistortion::s_StereoRigTransformationPropertyName).GetPointer());
    if (prop.IsNotNull())
    {
      itk::Matrix<float, 4, 4> txf = prop->GetValue();
      vtkSmartPointer<vtkMatrix4x4> tmpMatrix = vtkMatrix4x4::New();
      for (int i = 0; i < 4; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          tmpMatrix->SetElement(i, j, txf[i][j]);
        }
      }
      tmpMatrix->Invert();
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
      vtkSmartPointer<vtkMatrix4x4> trackingTransformMatrix = vtkMatrix4x4::New();
      trackingTransform->GetVtkMatrix(*trackingTransformMatrix);

      vtkSmartPointer<vtkMatrix4x4> combinedTransform = vtkMatrix4x4::New();
      vtkMatrix4x4::Multiply4x4(trackingTransformMatrix, m_TrackingCalibrationTransform, combinedTransform);

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
void QmitkSingle3DView::UpdateCameraToTrackImage()
{
  if (m_Image.IsNotNull())
  {
    int windowSize[2];
    windowSize[0] = this->width();
    windowSize[1] = this->height();

    int imageSize[2];
    imageSize[0] = m_Image->GetDimension(0);
    imageSize[1] = m_Image->GetDimension(1);

    double clippingRange[2];
    clippingRange[0] = m_ZNear;
    clippingRange[1] = m_ZFar;

    double origin[3];
    double spacing[3];
    double xAxis[3];
    double yAxis[3];

    mitk::Geometry3D::Pointer geometry = m_Image->GetGeometry();
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

    SetCameraParallelTo2DImage(imageSize, windowSize, origin, spacing, xAxis, yAxis, clippingRange, true, *m_MatrixDrivenCamera);
  }
}


