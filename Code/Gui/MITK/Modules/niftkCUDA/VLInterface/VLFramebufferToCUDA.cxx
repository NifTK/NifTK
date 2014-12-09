/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <VLInterface/VLFramebufferToCUDA.h>
#include <QGLWidget>
#include <boost/typeof/typeof.hpp>
#include <stdexcept>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>


//-----------------------------------------------------------------------------
VLFramebufferAdaptor::VLFramebufferAdaptor(vl::FramebufferObject* fbo, cudaStream_t stream)
  : m_GfxRes(0)
{
  assert(fbo->openglContext() != 0);

  // find the first colour attachment
  vl::ref<vl::FBOAbstractAttachment>    colorAttachment;
  for (int i = 0; i < (vl::AP_COLOR_ATTACHMENT15 - vl::AP_COLOR_ATTACHMENT0 + 1); ++i)
  {
    // some iterator type that derefs to FBOAbstractAttachment
    BOOST_AUTO(colorAttachmentIterator, fbo->fboAttachments().find(vl::EAttachmentPoint(vl::AP_COLOR_ATTACHMENT0 + i)));
    if (colorAttachmentIterator != fbo->fboAttachments().end())
    {
      colorAttachment = colorAttachmentIterator->second;
      break;
    }
  }

  if (colorAttachment.get() == 0)
  {
    throw std::runtime_error("FBO has no colour attachment");
  }

  // lets see if its a texture, or a renderbuffer
  bool    istexture       = dynamic_cast<vl::FBOTexture2DAttachment*>(colorAttachment.get())   != 0;
  bool    isrenderbuffer  = dynamic_cast<vl::FBOColorBufferAttachment*>(colorAttachment.get()) != 0;
  GLuint  glId      = 0;
  GLenum  glTarget  = 0;
  if (istexture)
  {
    glId = dynamic_cast<vl::FBOTexture2DAttachment*>(colorAttachment.get())->texture()->handle();
    glTarget = GL_TEXTURE_2D;
  }
  else
  {
    // sanity check
    if (!isrenderbuffer)
      throw std::runtime_error("FBO attachment is neither texture nor renderbuffer");

    glId = dynamic_cast<vl::FBOColorBufferAttachment*>(colorAttachment.get())->handle();
    glTarget = GL_RENDERBUFFER;
  }

  // FIXME: this needs a check to find out whether the ogl context for the fbo is currently active
  // FIXME: needs another check to find out which device that ogl context lives on

  cudaError_t             err = cudaSuccess;
  cudaGraphicsResource_t  gfxres = 0;
  err = cudaGraphicsGLRegisterImage(&gfxres, glId, glTarget, cudaGraphicsRegisterFlagsReadOnly);
  if (err != cudaSuccess)
  {
    throw std::runtime_error("Cannot register FBO attachment with CUDA");
  }

  err = cudaGraphicsMapResources(1, &gfxres, stream);
  if (err != cudaSuccess)
  {
    throw std::runtime_error("Cannot map FBO attachment into CUDA");
  }

  m_FBO = fbo;
  m_GfxRes = gfxres;
}


//-----------------------------------------------------------------------------
VLFramebufferAdaptor::~VLFramebufferAdaptor()
{
  // FIXME: docs dont say anything about synchronisation on unregister!
  //        only unmap is mentioned.
}


