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
#include <mitkDataStorage.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkOpenGLMatrixDrivenCamera.h>

namespace niftk
{
/**
 * \class SingleVideoWidget
 * \brief Derived from niftk::Single3DViewWidget to provide a widget that
 * given an image, will render it a video frame rate into either the
 * foreground or the background or both.
 *
 * This class can perform rendering, as if through a calibrated camera,
 * such as may be obtained view an OpenCV (Zhang 2000) camera model. If
 * the camera intrinsic parameters are found as a data node on the specified
 * image, then the camera switches to a calibrated camera, and otherwise
 * will assume the default VTK behaviour implemented in vtkOpenGLCamera.
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
   * \brief Sets a node, which should contain an mitk::CoordinateAxesData,
   * which is used for moving the camera around.
   *
   * For example, if this viewer is to render like a tracked laparoscope,
   * then this transform node should be the transform node from the tracker attached
   * to the laparoscope, and this class will then render a laparoscope viewpoint.
   */
  void SetTransformNode(const mitk::DataNode* node);

  /**
   * \brief Called from base class and gives us an opportunity to update renderings etc.
   */
  void Update();

protected:

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

  mitk::DataNode::Pointer                       m_TransformNode;
  vtkSmartPointer<vtkOpenGLMatrixDrivenCamera>  m_MatrixDrivenCamera;
  bool                                          m_IsCalibrated;
};

} // end namespace

#endif
