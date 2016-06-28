/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSingleVideoWidget_h
#define niftkSingleVideoWidget_h

#include "niftkIGIOverlayEditorExports.h"
#include "niftkSingle3DViewWidget.h"
#include "niftkBitmapOverlay.h"
#include <mitkDataStorage.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkOpenGLMatrixDrivenCamera.h>

namespace niftk
{
/**
 * \class SingleVideoWidget
 * \brief Derived from niftk::Single3DViewWidget to provide a widget that
 * given a video image, will render it at video frame rate into both the
 * foreground and the background layer, and additionally merge VTK renderings.
 *
 * According to VTK documentation, layers that are not the absolute background
 * should have a transparent background. So, by placing the VTK rendering in
 * a layer that is NOT the background, then the black background colour should
 * be transparent black. Then we also render the video into the foreground to try
 * and blend over it.
 *
 * In addition, this class can perform rendering, as if through a calibrated camera,
 * such as may be obtained view an OpenCV (Zhang 2000) camera model. If
 * the camera intrinsic parameters are found as a data node on the specified
 * image, then the camera switches to a calibrated camera, and otherwise
 * will assume the default VTK behaviour implemented in vtkOpenGLCamera.
 * This only does the intrinsics, not the distortion parameters. So the
 * supplied video should already be distortion corrected. The rendered
 * VTK world should be distortion corrected world.
 */
class NIFTKIGIOVERLAYEDITOR_EXPORT SingleVideoWidget : public Single3DViewWidget
{
  Q_OBJECT

public:

  SingleVideoWidget(QWidget* parent = 0,
                    Qt::WindowFlags f = 0,
                    mitk::RenderingManager* renderingManager = 0);

  virtual ~SingleVideoWidget();

  /**
   * \brief Stores ds in base-class and sets it on the niftk::BitmapOverlay.
   */
  virtual void SetDataStorage(mitk::DataStorage* ds);

  /**
   * \brief Overrides base class to set the selected node on the niftk::BitmapOverlay.
   */
  virtual void SetImageNode(mitk::DataNode* node);

  /**
   * \brief Called from base class and gives us an opportunity to update renderings etc.
   */
  virtual void Update();

  /**
   * \brief If true, will enable the bitmap overlay, if false, will disable it.
   */
  void SetUseOverlay(const bool& b);

  /**
   * \brief Sets a node, which should contain an mitk::CoordinateAxesData,
   * which is used for moving the camera around.
   *
   * For example, if this viewer is to render like a tracked laparoscope,
   * then this transform node should represent the world-to-camera
   * (i.e. tracker-to-hand, followed by hand-to-eye/camera) transform.
   */
  void SetTransformNode(const mitk::DataNode* node);

  /**
   * \brief Retrieves the opacity from the niftk::BitmapOverlayWidget.
   */
  float GetOpacity() const;

  /**
   * \brief Sets the opacity on the niftk::BitmapOverlayWidget.
   * \param value [0..1]
   */
  void SetOpacity(const float& value);

  /**
   * \brief If specified, will attempt to load the given file,
   * and will assume that the provided transform is a tracking matrix,
   * and calculate the camera-to-world transformation.
   */
  void SetEyeHandFileName(const std::string& fileName);

protected:

  /**
   * \brief Called when a DataStorage Node Removed Event was emitted.
   */
  virtual void NodeRemoved(const mitk::DataNode* node);

  /**
   * \brief Called when a DataStorage Node Changes Event was emitted.
   */
  virtual void NodeChanged(const mitk::DataNode* node);

  /**
   * \brief Called when a DataStorage Node Added Event was emitted.
   */
  virtual void NodeAdded(const mitk::DataNode* node);

  /**
   * \brief Re-implemented so we can correctly scale image while resizeing.
   */
  virtual void resizeEvent(QResizeEvent* event) override;

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

  niftk::BitmapOverlay::Pointer                 m_BitmapOverlay;
  mitk::DataNode::Pointer                       m_TransformNode;
  vtkSmartPointer<vtkOpenGLMatrixDrivenCamera>  m_MatrixDrivenCamera;
  bool                                          m_IsCalibrated;
  bool                                          m_UseOverlay;
  std::string                                   m_EyeHandFileName;
  vtkSmartPointer<vtkMatrix4x4>                 m_EyeHandMatrix;
};

} // end namespace

#endif
