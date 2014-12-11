/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SharedOGLContext.h"
#include <QMutexLocker>
#include <QGLContext>
#include <cassert>
#include <stdexcept>


//-----------------------------------------------------------------------------
QMutex                SharedOGLContext::s_Lock(QMutex::Recursive);
SharedOGLContext*     SharedOGLContext::s_Instance = 0;


//-----------------------------------------------------------------------------
SharedOGLContext::SharedOGLContext()
  : m_ShareWidget(0)
{
  // FIXME: find out virtual desktop size and make it span the full size so that
  //        our context would get assigned to every (enabled) gpu. maybe.
  m_ShareWidget = new QGLWidget(0, 0, Qt::WindowFlags(Qt::Window | Qt::FramelessWindowHint));

  if (!m_ShareWidget->context()->isValid())
  {
    delete m_ShareWidget;
    m_ShareWidget = 0;

    throw std::runtime_error("Cannot create OpenGL widget/context");
  }

  m_ShareWidget->hide();

  // run some benign opengl commands to see if the context is usable.
  // this is only ever going to fail if the driver is seriously broken.
  m_ShareWidget->makeCurrent();

  const GLubyte* str = glGetString(GL_RENDERER);
  // we don't care what it is. it should not error out, and not return null.
  if (glGetError() != GL_NO_ERROR)
  {
    m_ShareWidget->doneCurrent();
    delete m_ShareWidget;
    m_ShareWidget = 0;

    throw std::runtime_error("Cannot query most basic info from OpenGL context.");
  }
  if (str == 0)
  {
    m_ShareWidget->doneCurrent();
    delete m_ShareWidget;
    m_ShareWidget = 0;

    throw std::runtime_error("Cannot query most basic info from OpenGL context.");
  }

  m_ShareWidget->doneCurrent();
}


//-----------------------------------------------------------------------------
SharedOGLContext::~SharedOGLContext()
{
  delete m_ShareWidget;
}


//-----------------------------------------------------------------------------
QGLWidget* SharedOGLContext::GetShareWidget()
{
  QMutexLocker    lock(&s_Lock);

  if (s_Instance == 0)
  {
    s_Instance = new SharedOGLContext;
  }

  // the share-source should never be current!
  assert(QGLContext::currentContext() != s_Instance->m_ShareWidget->context());

  return s_Instance->m_ShareWidget;
}
