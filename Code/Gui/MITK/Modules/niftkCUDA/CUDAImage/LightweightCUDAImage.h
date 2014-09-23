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

#include "niftkCUDAExports.h"
#include <cuda_runtime_api.h>
#include <QAtomicInt>


// forward-decl
class CUDAManager;


/**
 * Wraps a device memory pointer with a bit of meta data and reference count.
 * Instances are created by CUDAManager.
 */
class NIFTKCUDA_EXPORT LightweightCUDAImage
{
  friend class CUDAManager;

public:
  /**
   * The default constructor creates an invalid object.
   * Only CUDAManager can create a valid one.
   */
  LightweightCUDAImage();

  /**
   * Non-virtual destructor: do not derive from this class.
   */
  ~LightweightCUDAImage();

  /** Copy constructor to handle reference count correctly. */
  LightweightCUDAImage(const LightweightCUDAImage& copyme);
  /** Assignment operator to handle reference count correctly. */
  LightweightCUDAImage& operator=(const LightweightCUDAImage& assignme);


  /**
   * Returns the ID of this image. Note that zero is not a valid ID!
   */
  unsigned int GetId() const;


  /**
   * Returns the CUDA event object that can be used to synchronise kernel
   * calls with completion of previous steps.
   * Use with cudaStreamWaitEvent().
   */
  cudaEvent_t GetReadyEvent() const;


private:
  QAtomicInt*     m_RefCount;

  int             m_Device;    // the device this image lives on.
  unsigned int    m_Id;
  cudaEvent_t     m_ReadyEvent;       // signaled when the image is ready for consumption.

  void*           m_DevicePtr;
  std::size_t     m_SizeInBytes;

  unsigned int    m_Width;          // in pixel
  unsigned int    m_Height;         // in (pixel) lines
  unsigned int    m_BytePitch;      // length of a line of pixels in bytes.
  // FIXME: pixel type descriptor
};

#endif // LightweightCUDAImage_h
