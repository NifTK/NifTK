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


//-----------------------------------------------------------------------------
QmitkVideoPreviewWidget::QmitkVideoPreviewWidget(QWidget* parent, QGLWidget* sharewith)
  : QGLWidget(parent, sharewith),
    m_TextureId(0), m_WidgetWidth(1), m_WidgetHeight(1), m_VideoWidth(1), m_VideoHeight(1)
{
  assert(this->isSharing());
}


//-----------------------------------------------------------------------------
void QmitkVideoPreviewWidget::SetVideoDimensions(int width, int height)
{
  m_VideoWidth  = std::max(width, 1);
  m_VideoHeight = std::max(height, 1);
}


//-----------------------------------------------------------------------------
void QmitkVideoPreviewWidget::SetTextureId(int id)
{
  m_TextureId = id;
}


//-----------------------------------------------------------------------------
void QmitkVideoPreviewWidget::initializeGL()
{
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  glClearColor(0, 0, 0, 0);

  assert(glGetError() == GL_NO_ERROR);
}


//-----------------------------------------------------------------------------
void QmitkVideoPreviewWidget::resizeGL(int width, int height)
{
  // dimensions can be smaller than zero (which would trigger an opengl error)
  m_WidgetWidth  = std::max(width, 1);
  m_WidgetHeight = std::max(height, 1);
}


//-----------------------------------------------------------------------------
void QmitkVideoPreviewWidget::setupViewport()
{
  // based on my videoapp standalone sdi recorder thingy

  // we assume square pixels coming from live camera
  //  (in encoded video this may be different: anamorphic, etc)
  float width_scale  = (float) m_WidgetWidth  / (float) m_VideoWidth;
  float height_scale = (float) m_WidgetHeight / (float) m_VideoHeight;

  int   vpw = m_WidgetWidth;
  int   vph = m_WidgetHeight;
  if (width_scale < height_scale)
  {
    vph = (int) ((float) m_VideoHeight * width_scale);
  }
  else
  {
    vpw = (int) ((float) m_VideoWidth * height_scale);
  }

  int vpx = m_WidgetWidth  / 2 - vpw / 2;
  int vpy = m_WidgetHeight / 2 - vph / 2;

  glViewport(vpx, vpy, vpw, vph);
  assert(glGetError() == GL_NO_ERROR);
}


//-----------------------------------------------------------------------------
void QmitkVideoPreviewWidget::paintGL()
{
  // do nothing if we dont have an up-to-date texture id
  if (m_TextureId == 0)
  {
    return;
  }

  setupViewport();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, m_TextureId);
  // we reset texture id so that we render at most one frame with it.
  // reason is that the sdi code uses a ringbuffer of textures
  // and using a repurposed texture id here causes some funny glitches.
  m_TextureId = 0;

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
  // dont leave it dangling
  glBindTexture(GL_TEXTURE_2D, 0);

  assert(glGetError() == GL_NO_ERROR);
}
