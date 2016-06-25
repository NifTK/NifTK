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
 * and scale the image to maximally fill the window.
 */
class NIFTKIGIOVERLAYEDITOR_EXPORT SingleUltrasoundWidget : public Single3DViewWidget
{
  Q_OBJECT

public:

  SingleUltrasoundWidget(QWidget* parent = 0,
                         Qt::WindowFlags f = 0,
                         mitk::RenderingManager* renderingManager = 0);

  virtual ~SingleUltrasoundWidget();

  void SetClipToImagePlane(const bool& clipToImagePlane);

  /**
   * \brief Called from base class and gives us an opportunity to update renderings etc.
   */
  virtual void Update();

private:

  /**
   * \brief Used to move the camera based on an image position and orientation,
   * such as might occur if you were using a tracked ultrasound image.
   *
   * In this case, we work out from the image size, and effective parallel projection.
   */
  void UpdateCameraToTrackImage();

  bool   m_ClipToImagePlane;
};

} // end namespace

#endif
