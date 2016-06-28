/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSingleUltrasoundWidget_h
#define niftkSingleUltrasoundWidget_h

#include "niftkIGIOverlayEditorExports.h"
#include "niftkSingle3DViewWidget.h"

namespace niftk
{
/**
 * \class SingleUltrasoundWidget
 * \brief Derived from niftk::Single3DViewWidget to provide a widget that
 * given an image, will always position the camera to face the image
 * and scale the image to maximally fill the window. In contrast to
 * niftk::SingleVideoWidget the image plane is a plane in 3D space,
 * not a projection through a camera model. Therefore, the image is
 * modelled as a texture plane, so 3D geometry objects can appear infront
 * or behind the plane.
 */
class NIFTKIGIOVERLAYEDITOR_EXPORT SingleUltrasoundWidget : public Single3DViewWidget
{
  Q_OBJECT

public:

  SingleUltrasoundWidget(QWidget* parent = 0,
                         Qt::WindowFlags f = 0,
                         mitk::RenderingManager* renderingManager = 0);

  virtual ~SingleUltrasoundWidget();

  /**
   * \brief Called from base class and gives us an opportunity to position camera.
   */
  virtual void Update();

  /**
   * \brief Overrides base class to create an additional texture plane mapper.
   * \see niftk::Single3DViewWidget::SetImageNode()
   */
  virtual void SetImageNode(mitk::DataNode* node);

  /**
   * \brief If true, sets the viewer to clip geometry +/- 1mm from image plane,
   * giving a crude contouring effect.
   */
  void SetClipToImagePlane(const bool& clipToImagePlane);

private:

  /**
   * \brief Used to move the camera based on an image position and orientation,
   * such as might occur if you were using a tracked ultrasound image.
   *
   * In this case, we work out from the image size, and assume parallel projection.
   */
  void UpdateCameraToTrackImage();

  /**
   * \brief Removes the texture plane mapper that this class manages.
   */
  void RemoveTextureMapper();

  // Turn on/off clipping. When off, clipping range is defined by
  // member variables in base class. When on, the camera is at a fixed
  // distance (1000mm) from the image plane, so we simply set +/- 1mm.
  bool   m_ClipToImagePlane;
};

} // end namespace

#endif
