/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkSingle3DView_h
#define QmitkSingle3DView_h

#include "niftkIGIGuiExports.h"
#include "QmitkBitmapOverlay.h"
#include <QmitkCmicLogo.h>
#include <mitkRenderWindowFrame.h>
#include <mitkGradientBackground.h>
#include <mitkDataStorage.h>
#include <QWidget>
#include <QFrame>
#include <QResizeEvent>
#include <QmitkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkOpenGLMatrixDrivenCamera.h>

class QGridLayout;
class QmitkRenderWindow;

namespace mitk {
  class RenderingManager;
}

/**
 * \class QmitkSingle3DView
 * \brief Widget containing a single 3D render window whose purpose
 * is to render a 3D VTK scene, blended with RGB image data, such as
 * might be available from video or ultrasound.
 *
 * This class was originally a stripped down QmitkStdMultiWidget, but
 * now it has been tidied up, it really is no longer a QmitkStdMultiWidget.
 *
 * This class can perform rendering, as if through a calibrated camera,
 * such as may be obtained view an OpenCV (Zhang 2000) camera model. If
 * the camera intrinsic parameters are found as a data node on the specified
 * image, then the camera switches to a calibrated camera, and otherwise
 * will assume the default VTK behaviour implemented in vtkOpenGLCamera.
 */
class NIFTKIGIGUI_EXPORT QmitkSingle3DView : public QWidget
{
  Q_OBJECT

public:

  QmitkSingle3DView(QWidget* parent = 0, Qt::WindowFlags f = 0, mitk::RenderingManager* renderingManager = 0);
  virtual ~QmitkSingle3DView();

  /**
   * \brief Stores ds locally, and sets the data storage on the contained
   * QmitkBitmapOverlay and QmitkRenderWindow.
   */
  void SetDataStorage( mitk::DataStorage* ds );

  /**
   * \brief Returns a pointer to the contained QmitkRenderWindow.
   */
  QmitkRenderWindow* GetRenderWindow() const;

  /**
   * \brief Retrieves the opacity from the QmitkBitmapOverlay.
   */
  float GetOpacity() const;

  /**
   * \brief Sets the opacity on the QmitkBitmapOverlay.
   * \param value [0..1]
   */
  void SetOpacity(const float& value);

  /**
   * \brief Passes the node onto the QmitkBitmapOverlay, so that
   * the QmitkBitmapOverlay can use it for a background or foreground image.
   * \param node an mitk::DataNode that should contain an RGB image.
   */
  void SetImageNode(const mitk::DataNode* node);

  /**
   * \brief Sets a node, which should contain an mitk::CoordinateAxesData,
   * which is used for moving the view-point around.
   *
   * For example, if this viewer is to render like a tracked laparoscope,
   * then this transform node should be the transform node from the tracker attached
   * to the laparoscope, and this class will then render a laparoscope viewpoint.
   */
  void SetTransformNode(const mitk::DataNode* node);

  /**
   * \brief Calls mitk::GradientBackground::EnableGradientBackground().
   */
  void EnableGradientBackground();

  /**
   * \brief Calls mitk::GradientBackground::DisableGradientBackground().
   */
  void DisableGradientBackground();

  /**
   * \brief Calls mitk::GradientBackground::SetGradientColors().
   */
  void SetGradientBackgroundColors( const mitk::Color & upper, const mitk::Color & lower );

  /**
   * \brief Calls mitk::CMICLogo::EnableDepartmentLogo(), and is currently unused.
   */
  void EnableDepartmentLogo();

  /**
   * \brief Calls mitk::CMICLogo::DisableDepartmentLogo(), and is currently unused.
   */
  void DisableDepartmentLogo();

  /**
   * \brief Calls mitk::CMICLogo::SetDepartmentLogoPath(), and is currently unused.
   */
  void SetDepartmentLogoPath( const char * path );

  /**
   * \brief Sets the Tracking Calibration file name, which causes a re-loading of the tracking calibration matrix.
   */
  void SetTrackingCalibrationFileName(const std::string& fileName);

  /**
   * \brief Sets whether or not we are doing camera tracking mode.
   *
   * For Video work, we set this to true, and the video image in expected
   * to have the camera intrinsics attached as a property. In addition, the camera
   * position is transformed using the combo box. The end result is that the
   * VTK objects should overlay the displayed video image according to a calibrated
   * camera projection.
   *
   * For ultrasound work, we set this to false. In this mode, the image is simply
   * presented, and a parallel projection mode is used to render the VTK objects
   * on top of the image.
   */
  void SetCameraTrackingMode(const bool& isCameraTracking);

  /**
   * \brief Called from QmitkIGIOverlayEditor to indicate that transformations should all be updated.
   */
  void Update();

protected:

  /**
   * \brief Re-implemented so we can tell QmitkBitmapOverlay the display size has changed.
   */
  virtual void resizeEvent(QResizeEvent* event);

private:

  /**
   * \brief Separate method, so we can force an update on each refresh.
   */
  void UpdateCameraIntrinsicParameters();

  /**
   * \brief Used to move the camera based on a tracking transformation.
   *
   * In addition, we also have a fallback position. If the camera calibration
   * information is not found, we switch the camera to be a parallel projection,
   * based on the size of the image, but still move the camera when a tracking
   * transformation and calibration transformation are available.
   */
  void UpdateCameraViaTrackingTransformation();

  /**
   * \brief Used to move the camera based on an image position and orientation,
   * such as might occur if you were using a tracked ultrasound image.
   *
   * In this case, we work out from the image size, and effective parallel projection.
   */
  void UpdateCameraToTrackImage();

  /**
   * \brief Utility method to deregister data storage listeners.
   */
  void DeRegisterDataStorageListeners();

  /**
   * \brief Called when a DataStorage Node Removed Event was emitted.
   */
  void NodeRemoved(const mitk::DataNode* node);

  /**
   * \brief Called when a DataStorage Node Changed Event was emitted.
   */
  void NodeChanged(const mitk::DataNode* node);

  /**
   * \brief Called when a DataStorage Node Added Event was emitted.
   */
  void NodeAdded(const mitk::DataNode* node);

  mitk::DataStorage::Pointer                    m_DataStorage;
  QmitkRenderWindow                            *m_RenderWindow;
  QGridLayout                                  *m_Layout;
  mitk::RenderingManager                       *m_RenderingManager;
  mitk::RenderWindowFrame::Pointer              m_RenderWindowFrame;
  mitk::GradientBackground::Pointer             m_GradientBackground;
  CMICLogo::Pointer                             m_LogoRendering;
  QmitkBitmapOverlay::Pointer                   m_BitmapOverlay;
  std::string                                   m_TrackingCalibrationFileName;
  vtkSmartPointer<vtkMatrix4x4>                 m_TrackingCalibrationTransform;
  mitk::DataNode::Pointer                       m_TransformNode;
  mitk::Image::Pointer                          m_Image;
  mitk::DataNode::Pointer                       m_ImageNode;
  vtkSmartPointer<vtkOpenGLMatrixDrivenCamera>  m_MatrixDrivenCamera;
  bool                                          m_IsCameraTracking;
  bool                                          m_IsCalibrated;
  double                                        m_ZNear;
  double                                        m_ZFar;
};
#endif /* QmitkSingle3DView */
