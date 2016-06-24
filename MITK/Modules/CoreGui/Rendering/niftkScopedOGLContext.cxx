/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkScopedOGLContext.h"
#include <QGLContext>
#include <cassert>
#include <stdexcept>

namespace niftk
{

//-----------------------------------------------------------------------------
ScopedOGLContext::ScopedOGLContext(QGLContext* newctx)
  : m_Ourctx(newctx)
{
  m_Prevctx = const_cast<QGLContext*>(QGLContext::currentContext());
  m_Ourctx->makeCurrent();
}


//-----------------------------------------------------------------------------
ScopedOGLContext::~ScopedOGLContext()
{
  // did somebody mess up our context?
  assert(QGLContext::currentContext() == m_Ourctx);

  if (m_Prevctx)
    m_Prevctx->makeCurrent();
  else
    m_Ourctx->doneCurrent();
}

} // end namespace
