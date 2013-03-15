/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cassert>
#include <QGLWidget>
#include "QmitkVideoPreviewWidget.h"


 QmitkVideoPreviewWidget::QmitkVideoPreviewWidget(QWidget* parent, QGLWidget* sharewith)
  : QGLWidget(parent, sharewith),
    textureid(0)
{
  assert(this->isSharing());
}

void QmitkVideoPreviewWidget::set_video_dimensions(int width, int height)
{
}

void QmitkVideoPreviewWidget::set_texture_id(int id)
{
  textureid = id;
}

void QmitkVideoPreviewWidget::initializeGL()
{
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  assert(glGetError() == GL_NO_ERROR);
}

void QmitkVideoPreviewWidget::resizeGL(int width, int height)
{
  // dimensions can be smaller than zero (which would trigger an opengl error)
  width  = std::max(width, 1);
  height = std::max(height, 1);
  glViewport(0, 0, width, height);

  assert(glGetError() == GL_NO_ERROR);
}

void QmitkVideoPreviewWidget::paintGL()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, textureid);

  glColor4f(1, 1, 1, 1);
  glBegin(GL_QUADS);
    glTexCoord2f(0, 1);
    glVertex2f(-1,  1);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f( 1, -1);
    glTexCoord2f(1, 1);
    glVertex2f( 1,  1);
  glEnd();

  assert(glGetError() == GL_NO_ERROR);
}
