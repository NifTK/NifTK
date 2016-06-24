/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkScopedOGLContext_h
#define niftkScopedOGLContext_h

#include <niftkCoreGuiExports.h>
#include <QtOpenGL/QGLContext>

namespace niftk
{

/**
* \brief Utility class to activate a given OpenGL context,
* and restore the previous one on scope-exit.
*/
struct NIFTKCOREGUI_EXPORT ScopedOGLContext
{
  ScopedOGLContext(QGLContext* newctx);
  ~ScopedOGLContext();

  QGLContext*   m_Prevctx;
  QGLContext*   m_Ourctx;
};

} // end namespace

#endif // niftkScopedOGLContext_h
