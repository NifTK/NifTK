/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __ImageInfoRenderer_h
#define __ImageInfoRenderer_h

#include <QObject>
#include "PluginCore.h"

class ImageInfoRendererPrivate;
class QmitkRenderWindow;
class QString;

namespace mitk {
class DataNode;
}

class ImageInfoRenderer : public QObject, public PluginCore
{
  Q_OBJECT

public:

  explicit ImageInfoRenderer();
  virtual ~ImageInfoRenderer();

  virtual void onNodeAdded(const mitk::DataNode* node);
  virtual void onNodeRemoved(const mitk::DataNode* node);
  virtual void onVisibilityChanged(const mitk::DataNode* node);


  void DisplayImageInfo(const QString& text);
  void DisplayImageInfo(const QString& text, int i);

public slots:
  ///
  /// Called when the renderwindow gets deleted
  ///
  void OnRenderWindowDelete(QObject * obj = 0);

private slots:
  void onCrosshairVisibilityChanged(const mitk::DataNode* crosshairNode);
  void onCrosshairPositionEventDelayed();
  void showPixelValue(mitk::Point3D crosshairPos);

private:
  void onCrosshairPositionEvent();

  QScopedPointer<ImageInfoRendererPrivate> d_ptr;

  Q_DECLARE_PRIVATE(ImageInfoRenderer);
  Q_DISABLE_COPY(ImageInfoRenderer);
};

#endif
