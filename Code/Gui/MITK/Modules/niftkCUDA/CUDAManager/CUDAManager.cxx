/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "CUDAManager.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <QMutexLocker>
#include <cassert>
#include <cmath>
#include <new>


//-----------------------------------------------------------------------------
static bool CUDADelayLoadCheck()
{
  // the cuda dlls are delay-loaded, that means they are only mapped into our process
  // on demand, i.e. when we try to call an exported function for the first time.
  // this is what we do here.
  // if it fails (e.g. dll not found) then the runtime linker will throw a SEH exception.
  __try
  {
    // touch an entry point in nvcuda.dll
    int   driverversion = 0;
    CUresult r = cuDriverGetVersion(&driverversion);
    // touch an entry point in cudart*.dll
    int   runtimeversion = 0;
    cudaError_t s = cudaRuntimeGetVersion(&runtimeversion);

    // we dont care about the result.
    // (actually, it would make sense to check that the driver supports cuda 5).

    return true;
  }
  __except(1)
  {
    return false;
  }

  // unreachable
  assert(false);
}


//-----------------------------------------------------------------------------
CUDAManager*      CUDAManager::s_Instance = 0;


//-----------------------------------------------------------------------------
CUDAManager* CUDAManager::GetInstance()
{
  if (s_Instance == 0)
  {
    bool  ok = CUDADelayLoadCheck();
    if (!ok)
    {
      throw std::runtime_error("No CUDA available. Delay-load check failed.");
    }

    s_Instance = new CUDAManager;
  }


  return s_Instance;
}


//-----------------------------------------------------------------------------
CUDAManager::CUDAManager()
  : m_Lock(QMutex::Recursive)
  // zero is not a valid id, but it's good for init: we'll inc this one and the first valid id is then 1.
  , m_LastIssuedId(0)
{

}


//-----------------------------------------------------------------------------
CUDAManager::~CUDAManager()
{
}


//-----------------------------------------------------------------------------
cudaStream_t CUDAManager::GetStream(const std::string& name)
{
  QMutexLocker    lock(&m_Lock);

  std::map<std::string, cudaStream_t>::const_iterator i = m_Streams.find(name);
  if (i == m_Streams.end())
  {
    cudaStream_t    stream;
    cudaError_t     err;
    // cudaStreamNonBlocking means that there is no implicit synchronisation to the null stream.
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // FIXME: error handling?
    assert(err == cudaSuccess);

    i = m_Streams.insert(std::make_pair(name, stream)).first;
  }

  return i->second;
}


//-----------------------------------------------------------------------------
WriteAccessor CUDAManager::RequestOutputImage(unsigned int width, unsigned int height, int FIXME_pixeltype)
{
  QMutexLocker    lock(&m_Lock);

  // FIXME: figure out how to best deal with pixel types.

  // round up the length of a line of pixels to make sure access is aligned.
  unsigned int  bytepitch = width * FIXME_pixeltype;
  // 64 byte alignment sounds good.
  bytepitch += bytepitch % 64;
  assert((bytepitch % 64) == 0);

  std::size_t   minSizeInBytes = bytepitch * height;

  unsigned int   sizeTier = SizeToTier(minSizeInBytes);
  if (m_ImagePool.size() <= sizeTier)
  {
    m_ImagePool.resize(sizeTier + 1);
  }

  std::list<LightweightCUDAImage>::iterator i = m_ImagePool[sizeTier].begin();
  if (i == m_ImagePool[sizeTier].end())
  {
    // the amount we request from the driver is always the full tier size.
    // makes management simpler, but wastes memory (we got plenty though).
    std::size_t   fullTierSize = TierToSize(sizeTier);
    assert(fullTierSize >= minSizeInBytes);

    cudaError_t   err = cudaSuccess;
    void*         devptr = 0;
    err = cudaMalloc(&devptr, fullTierSize);
    if (err == cudaErrorMemoryAllocation)
    {
      throw std::bad_alloc();
    }
    assert(err == cudaSuccess);

    ++m_LastIssuedId;
    // zero is an invalid id, so if we overflow into zero then bad things will happen.
    assert(m_LastIssuedId != 0);

    LightweightCUDAImage  lwci;
    lwci.m_Id         = m_LastIssuedId;
    lwci.m_DevicePtr  = devptr;
    lwci.m_Width      = width;
    lwci.m_Height     = height;
    lwci.m_BytePitch  = bytepitch;

    // the ready-event is specifically not for timing, only for stream-sync.
    err = cudaEventCreateWithFlags(&lwci.m_ReadyEvent, cudaEventDisableTiming);
    if (err != cudaSuccess)
    {
      err = cudaFree(devptr);
      assert(err == cudaSuccess);
      throw std::runtime_error("Cannot create CUDA event");
    }

    i = m_ImagePool[sizeTier].insert(m_ImagePool[sizeTier].begin(), lwci);
  }

  m_InFlightOutputImages.insert(std::make_pair(i->m_Id, *i));

  WriteAccessor   wa;
  wa.m_Id             = i->m_Id;
  wa.m_ReadyEvent     = i->m_ReadyEvent;
  wa.m_DevicePointer  = i->m_DevicePtr;
  wa.m_SizeInBytes    = TierToSize(sizeTier);

  return wa;
}


//-----------------------------------------------------------------------------
std::size_t CUDAManager::TierToSize(unsigned int tier) const
{
  return 1 << tier;
}


//-----------------------------------------------------------------------------
unsigned int CUDAManager::SizeToTier(std::size_t size) const
{
  // essentially log2 for integer
  unsigned int ret = 0;
  while (size != 0)
  {
    size >>= 1;
    ++ret;
  }
  return ret;
}


//-----------------------------------------------------------------------------
LightweightCUDAImage CUDAManager::Finalise(WriteAccessor& writeAccessor, cudaStream_t stream)
{
  QMutexLocker    lock(&m_Lock);

  std::map<unsigned int, LightweightCUDAImage>::iterator i = m_InFlightOutputImages.find(writeAccessor.m_Id);
  if (i == m_InFlightOutputImages.end())
  {
    throw std::runtime_error("Invalid WriteAccessor passed to Finalise()");
  }
  // sanity check
  assert(i->second.GetId() == i->first);

  LightweightCUDAImage  lwci = i->second;
  m_InFlightOutputImages.erase(i);

  cudaError_t   err = cudaSuccess;
  err = cudaEventRecord(lwci.m_ReadyEvent, stream);
  assert(err == cudaSuccess);

  // queue completion callback?


  bool inserted = m_ValidImages.insert(std::make_pair(lwci.GetId(), lwci)).second;
  assert(inserted);

  // invalidate writeAccessor
  writeAccessor.m_DevicePointer = 0;

  return lwci;
}
