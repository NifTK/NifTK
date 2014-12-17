/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "ScopedOGLContext.h"
#include <QGLContext>
#include <cassert>
#include <stdexcept>


//-----------------------------------------------------------------------------
ScopedOGLContext::ScopedOGLContext(QGLContext* newctx)
  : ourctx(newctx)
{
  prevctx = const_cast<QGLContext*>(QGLContext::currentContext());
  if (prevctx != ourctx)
    ourctx->makeCurrent();
}


//-----------------------------------------------------------------------------
ScopedOGLContext::~ScopedOGLContext()
{
  // did somebody mess up our context?
  assert(QGLContext::currentContext() == ourctx);

  if (prevctx != ourctx)
  {
    if (prevctx)
      prevctx->makeCurrent();
    else
      ourctx->doneCurrent();
  }
}
