/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "FlipImage.h"
#include <Image/FlipImageKernel.h>
#include <stdexcept>


//-----------------------------------------------------------------------------
void FlipImage(const WriteAccessor& src, WriteAccessor& dest, cudaStream_t stream)
{
  if (dest.m_SizeInBytes < src.m_SizeInBytes)
    throw std::runtime_error("Output buffer is smaller than input");

  int     widthInBytes  = std::min(dest.m_PixelWidth, src.m_PixelWidth) * dest.m_FIXME_pixeltype;
  int     height        = std::min(dest.m_PixelHeight, src.m_PixelHeight);

  RunFlipImageKernel(
      (char*) dest.m_DevicePointer,
      widthInBytes,
      height,
      dest.m_BytePitch,
      (char*) src.m_DevicePointer,
      src.m_BytePitch,
      stream
  );
}
