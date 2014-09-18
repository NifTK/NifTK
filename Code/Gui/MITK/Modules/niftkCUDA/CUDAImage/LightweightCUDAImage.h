/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#ifndef LightweightCUDAImage_h
#define LightweightCUDAImage_h

#include <cuda_runtime_api.h>

// forward-decl
class CUDAManager;


class LightweightCUDAImage
{
  friend class CUDAManager;

public:
  LightweightCUDAImage()
    : m_Id(0)
  {
  }


  // zero is not a valid id.
  unsigned int GetId() const
  {
    return m_Id;
  }


private:
  int             m_Device;    // the device this image lives on.
  unsigned int    m_Id;
  cudaEvent_t     m_ReadyEvent;       // signaled when the image is ready for consumption.

  void*           m_DevicePtr;

  unsigned int    m_Width;          // in pixel
  unsigned int    m_Height;         // in (pixel) lines
  unsigned int    m_BytePitch;      // length of a line of pixels in bytes.
  // FIXME: pixel type descriptor

};

#endif // LightweightCUDAImage_h
