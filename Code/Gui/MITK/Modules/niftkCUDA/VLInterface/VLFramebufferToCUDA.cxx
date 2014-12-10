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
#include <cassert>
#include <stdexcept>
#include <cuda_gl_interop.h>
#include <mitkLogMacros.h>
#include <vlGraphics/OpenGLContext.hpp>


//-----------------------------------------------------------------------------
VLFramebufferAdaptor::VLFramebufferAdaptor(vl::FramebufferObject* fbo)
  : m_GfxRes(0)
{
  assert(fbo->openglContext() != 0);

  // FIXME: i'd prefer to leave this to the caller and instead check/assert here.
  //        but there is currently no api for that in vl.
  fbo->openglContext()->makeCurrent();

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


  m_FBO = fbo;
  m_GfxRes = gfxres;
}


//-----------------------------------------------------------------------------
VLFramebufferAdaptor::~VLFramebufferAdaptor()
{
  cudaError_t   err = cudaSuccess;

  // check if resource has been unmapped.
  // it's an error not to unmap first.
  {
    cudaArray_t   arr = 0;
    err = cudaGraphicsSubResourceGetMappedArray(&arr, m_GfxRes, 0, 0);
    // docs say: "If resource is not mapped then cudaErrorUnknown is returned."
    if (err != cudaErrorUnknown)
    {
      MITK_WARN << "Forgot to call VLFramebufferAdaptor::Unmap()";
      // in debug mode die hard.
      assert(!"Forgot to call VLFramebufferAdaptor::Unmap()");

      try
      {
        // this is unsafe: we dont know which stream is still using it.
        Unmap(0);
      }
      catch (...)
      {
        MITK_WARN << "Double-fault on missed VLFramebufferAdaptor::Unmap() cleanup";
      }
    }
  }

  // FIXME: docs dont say anything about synchronisation on unregister!
  //        only unmap is mentioned.

  err = cudaGraphicsUnregisterResource(m_GfxRes);
  if (err != cudaSuccess)
  {
    MITK_WARN << "Failed to unregister FBO attachment from CUDA";
    // die in debug mode.
    assert(!"Failed to unregister FBO attachment from CUDA");
  }
}


//-----------------------------------------------------------------------------
cudaArray_t VLFramebufferAdaptor::Map(cudaStream_t stream)
{
  cudaError_t   err = cudaSuccess;
  err = cudaGraphicsMapResources(1, &m_GfxRes, stream);
  if (err != cudaSuccess)
  {
    throw std::runtime_error("Cannot map FBO attachment into CUDA");
  }

  cudaArray_t   arr = 0;
  err = cudaGraphicsSubResourceGetMappedArray(&arr, m_GfxRes, 0, 0);
  if (err != cudaSuccess)
  {
    try
    {
      Unmap(stream);
    }
    catch (...)
    {
      MITK_WARN << "double-fault cuda-mapping/unmapping. ignoring it.";
    }

    throw std::runtime_error("Cannot get CUDA arry for FBO attachment");
  }

  return arr;
}


//-----------------------------------------------------------------------------
void VLFramebufferAdaptor::Unmap(cudaStream_t stream)
{
  cudaError_t   err = cudaSuccess;
  err = cudaGraphicsUnmapResources(1, &m_GfxRes, stream);
  if (err != cudaSuccess)
  {
    throw std::runtime_error("Cannot unmap array of FBO attachment");
  }
}
