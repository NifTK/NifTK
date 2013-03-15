/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKVIDEOPREVIEWWIDGET_H
#define QMITKVIDEOPREVIEWWIDGET_H

#include <QObject>
#include <QMetaType>
#include <QGLWidget>


class QmitkVideoPreviewWidget : public QGLWidget
{
  Q_OBJECT

public:
  QmitkVideoPreviewWidget(QWidget* parent, QGLWidget* sharewith);


public:
  void set_video_dimensions(int width, int height);
  void set_texture_id(int id);


protected:
  virtual void initializeGL();
  virtual void resizeGL(int width, int height);
  virtual void paintGL();

  int   textureid;
};

#endif // QMITKVIDEOPREVIEWWIDGET_H
