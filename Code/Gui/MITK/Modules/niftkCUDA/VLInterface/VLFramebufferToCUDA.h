/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef VLFramebufferToCUDA_h
#define VLFramebufferToCUDA_h

#include "niftkCUDAExports.h"
#include <vlGraphics/FramebufferObject.hpp>
#include <driver_types.h>


class NIFTKCUDA_EXPORT VLFramebufferAdaptor
{

public:
  VLFramebufferAdaptor(vl::FramebufferObject* fbo, cudaStream_t stream);
  ~VLFramebufferAdaptor();


private:
  vl::ref<vl::FramebufferObject>      m_FBO;
  cudaGraphicsResource_t              m_GfxRes;
};



#endif // VLFramebufferToCUDA
