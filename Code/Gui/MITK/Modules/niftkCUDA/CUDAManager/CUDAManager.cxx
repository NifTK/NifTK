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
#ifdef _MSC_VER
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

#else
  return true;
#endif
}


//-----------------------------------------------------------------------------
QMutex            CUDAManager::s_Lock(QMutex::Recursive);
CUDAManager*      CUDAManager::s_Instance = 0;


namespace impldetail
{

/** Static initialiser to cleanup on module unload. */
struct ModuleCleanup
{
  ModuleCleanup()
  {
    // nothing to do.
  }

  ~ModuleCleanup()
  {
    try
    {
      // see https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/3977 for why this is disabled for now.
#if 0
      delete CUDAManager::s_Instance;
#endif
      CUDAManager::s_Instance = 0;
    }
    catch (...)
    {
      // swallow it. never let exceptions escape from destructors.
    }
  }
};

} // namespace

// remember: objects at file scope are constructed and cleared up in the order of declaration.
// so this one will be done before the lock above goes away.
impldetail::ModuleCleanup       s_ModuleCleaner;


/**
 * @internal
 * Used by FinaliseAndAutorelease() and StreamCallback() to pass information through
 * the CUDA driver about what should be released.
 */
struct StreamCallbackReleasePOD
{
  CUDAManager*        m_Manager;
  unsigned int        m_Id;
};


//-----------------------------------------------------------------------------
CUDAManager* CUDAManager::GetInstance()
{
  QMutexLocker    lock(&s_Lock);

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
  // zero is not a valid id, but it's good for init: we'll inc this one and the first valid id is then 1.
  : m_LastIssuedId(0)
{

}


//-----------------------------------------------------------------------------
CUDAManager::~CUDAManager()
{
  for (std::map<std::string, cudaStream_t>::iterator i = m_Streams.begin(); i != m_Streams.end(); ++i)
  {
    cudaError_t err;
    err = cudaStreamDestroy(i->second);
    assert(err == cudaSuccess);
  }


  // these should be empty! but depending on how other code uses datanodes etc it could
  // leak a reference which then means that we also leak one here.
  // in that case, simply clean up explicitly.
  // that would make the LightweightCUDAImage instance still floating around invalid but hey too bad:
  // the runtime lib is being unloaded, don't expect anything to work afterwards.

  for (std::map<unsigned int, LightweightCUDAImage>::iterator i = m_ValidImages.begin(); i != m_ValidImages.end(); ++i)
  {
    // remember this odd one: the LightweightCUDAImage that sits on a DataNode increments the refcount for it.
    // but that LightweightCUDAImage is only a handle, so we need to keep it here as well to manage it.
    // images that are referenced somewhere sit in m_ValidImages, but to prevent that from never moving them
    // back to m_AvailableImagePool, images in m_ValidImages do not have their refcount increased.

    // instead of just inc'ing the refcount and leaving the cleanup for m_AvailableImagePool below deal with it,
    // we free these explicitly to avoid recursive calls via AllRefsDropped() into mutative operations on m_ValidImages.

    // FIXME: make sure to activate the correct device context.

    cudaError_t err;
    err = cudaEventDestroy(i->second.m_ReadyEvent);
    assert(err == cudaSuccess);
    i->second.m_ReadyEvent = 0;

    err = cudaFree(i->second.m_DevicePtr);
    assert(err == cudaSuccess);
    i->second.m_DevicePtr = 0;

    delete i->second.m_RefCount;
    i->second.m_RefCount = 0;
  }
  // because the refcount for all images in m_ValidImages is gone now, this will not
  // trigger a call to AllRefsDropped() (which would place them on m_AvailableImagePool correctly but) that
  // double-modifies m_ValidImages and trashes its internal state leading to a crash.
  m_ValidImages.clear();

  //std::map<unsigned int, LightweightCUDAImage>    m_InFlightOutputImages


  for (int i = 0; i < m_AvailableImagePool.size(); ++i)
  {
    for (std::list<LightweightCUDAImage>::iterator j = m_AvailableImagePool[i].begin(); j != m_AvailableImagePool[i].end(); ++j)
    {
      // FIXME: make sure to activate the correct device context.

      cudaError_t err;
      err = cudaEventDestroy(j->m_ReadyEvent);
      assert(err == cudaSuccess);
      j->m_ReadyEvent = 0;

      err = cudaFree(j->m_DevicePtr);
      assert(err == cudaSuccess);
      j->m_DevicePtr = 0;

      delete j->m_RefCount;
      j->m_RefCount = 0;
    }
  }
  // explicitly clear so we are independent of class member destruction order.
  // (that sounds a bit like a bug...)
  m_AvailableImagePool.clear();
}


//-----------------------------------------------------------------------------
cudaStream_t CUDAManager::GetStream(const std::string& name)
{
  QMutexLocker    lock(&s_Lock);

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
  QMutexLocker    lock(&s_Lock);

  // FIXME: figure out how to best deal with pixel types.

  // round up the length of a line of pixels to make sure access is aligned.
  unsigned int  bytepitch = width * FIXME_pixeltype;
  bytepitch = std::max(bytepitch, 64u);
  // 64 byte alignment sounds good.
  bytepitch = ((bytepitch / 64) + 1) * 64;
  assert((bytepitch % 64) == 0);

  std::size_t   minSizeInBytes = bytepitch * height;

  unsigned int   sizeTier = SizeToTier(minSizeInBytes);
  if (m_AvailableImagePool.size() <= sizeTier)
  {
    m_AvailableImagePool.resize(sizeTier + 1);
  }

  std::list<LightweightCUDAImage>::iterator i = m_AvailableImagePool[sizeTier].begin();
  if (i == m_AvailableImagePool[sizeTier].end())
  {
    // the amount we request from the driver is always the full tier size.
    // makes management simpler, but wastes memory (we got plenty though).
    std::size_t   fullTierSize = TierToSize(sizeTier);
    assert(fullTierSize >= minSizeInBytes);
    assert(SizeToTier(fullTierSize) == sizeTier);

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
    lwci.m_RefCount   = new QAtomicInt(1);
    lwci.m_Id         = m_LastIssuedId;
    lwci.m_DevicePtr  = devptr;
    lwci.m_Width      = width;
    lwci.m_Height     = height;
    lwci.m_BytePitch  = bytepitch;
    lwci.m_SizeInBytes = fullTierSize;

    // the ready-event is specifically not for timing, only for stream-sync.
    err = cudaEventCreateWithFlags(&lwci.m_ReadyEvent, cudaEventDisableTiming);
    if (err != cudaSuccess)
    {
      err = cudaFree(devptr);
      assert(err == cudaSuccess);
      throw std::runtime_error("Cannot create CUDA event");
    }

    i = m_AvailableImagePool[sizeTier].insert(m_AvailableImagePool[sizeTier].begin(), lwci);
  }
  else
  {
    // images in m_AvailableImagePool are not referenced by anyone else anymore.
    // so these should get a new id.
    ++m_LastIssuedId;
    i->m_Id = m_LastIssuedId;
  }

  bool inserted = m_InFlightOutputImages.insert(std::make_pair(i->m_Id, *i)).second;
  assert(inserted);

  WriteAccessor   wa;
  wa.m_Id             = i->m_Id;
  wa.m_ReadyEvent     = i->m_ReadyEvent;
  wa.m_DevicePointer  = i->m_DevicePtr;
  wa.m_SizeInBytes    = i->m_SizeInBytes;
  wa.m_BytePitch      = i->m_BytePitch;
  // the to be returned WriteAccessor has an implicit reference to the image.
  // so keep it alive.
  i->m_RefCount->ref();

  // if the image is used for output then it is no longer available for something else.
  m_AvailableImagePool[sizeTier].erase(i);

  return wa;
}


//-----------------------------------------------------------------------------
std::size_t CUDAManager::TierToSize(unsigned int tier) const
{
  return (1 << tier) - 1;
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
  QMutexLocker    lock(&s_Lock);

  std::map<unsigned int, LightweightCUDAImage>::iterator i = m_InFlightOutputImages.find(writeAccessor.m_Id);
  if (i == m_InFlightOutputImages.end())
  {
    throw std::runtime_error("Invalid WriteAccessor passed to Finalise()");
  }
  // sanity check
  assert(i->second.GetId() == i->first);

  LightweightCUDAImage  lwci = i->second;

  cudaError_t   err = cudaSuccess;
  err = cudaEventRecord(lwci.m_ReadyEvent, stream);
  assert(err == cudaSuccess);

  bool inserted = m_ValidImages.insert(std::make_pair(lwci.GetId(), lwci)).second;
  assert(inserted);
  // important to update the two lists (m_ValidImages and m_InFlightOutputImages) close together so
  // that we can handle exceptions properly. say if insert fails for one we could otherwise leak memory.
  m_InFlightOutputImages.erase(i);

  // invalidate writeAccessor
  writeAccessor.m_Id = 0;
  writeAccessor.m_DevicePointer = 0;
  // deref because writeaccessor has now lost its implicit reference to this image.
  lwci.m_RefCount->deref();

  // while being held by m_ValidImages, refs in there dont count towards the reference count.
  // this is so that m_ValidImages does not keep them artificially alive.
  // i.e. their refcount would never drop to zero if all datastorage nodes are gone, hence
  // they'd never end up back on the m_AvailableImagePool.
  lwci.m_RefCount->deref();

  return lwci;
}


//-----------------------------------------------------------------------------
ReadAccessor CUDAManager::RequestReadAccess(const LightweightCUDAImage& lwci)
{
  QMutexLocker    lock(&s_Lock);

  std::map<unsigned int, LightweightCUDAImage>::const_iterator i = m_ValidImages.find(lwci.GetId());
  if (i == m_ValidImages.end())
  {
    throw std::runtime_error("Requested image is not (yet?) valid");
  }

  ReadAccessor  ra;
  ra.m_Id             = lwci.m_Id;
  ra.m_DevicePointer  = lwci.m_DevicePtr;
  ra.m_ReadyEvent     = lwci.m_ReadyEvent;
  ra.m_SizeInBytes    = lwci.m_SizeInBytes;
  ra.m_BytePitch      = lwci.m_BytePitch;
  // readaccessor has an implicit ref to the image.
  i->second.m_RefCount->ref();

  return ra;
}


//-----------------------------------------------------------------------------
LightweightCUDAImage CUDAManager::FinaliseAndAutorelease(WriteAccessor& writeAccessor, ReadAccessor& readAccessor, cudaStream_t stream)
{
  QMutexLocker    lock(&s_Lock);

  LightweightCUDAImage  lwci = Finalise(writeAccessor, stream);

  // it's important that release comes after finalise!
  // otherwise, the ready-event will be blocked by the stream-callback.

  Autorelease(readAccessor, stream);

  return lwci;
}


//-----------------------------------------------------------------------------
void CUDAManager::Autorelease(ReadAccessor& readAccessor, cudaStream_t stream)
{
  QMutexLocker    lock(&s_Lock);

  // the image represented by readAccessor can only be released once the kernel has finished with it.
  // queueing a callback onto the stream will tell us when.
  StreamCallbackReleasePOD*   pod = new StreamCallbackReleasePOD;
  pod->m_Manager = this;
  pod->m_Id      = readAccessor.m_Id;

  cudaError_t   err = cudaSuccess;
  err = cudaStreamAddCallback(stream, StreamCallback, pod, 0);
  if (err != cudaSuccess)
  {
    // this is a critical error: we wont be able to cleanup the refcount for read-requested-images.
    delete pod;
    throw std::runtime_error("Cannot queue stream callback");
  }


  // invalidate readaccessor.
  // to be done immediately, completion callback would be too late!
  readAccessor.m_Id = 0;
  readAccessor.m_DevicePointer = 0;
}


//-----------------------------------------------------------------------------
void CUDAManager::AllRefsDropped(LightweightCUDAImage& lwci, bool fromStreamCallback)
{
  QMutexLocker    lock(&s_Lock);

  // by definition, lwci can not be on m_InFlightOutputImages.
  assert(m_InFlightOutputImages.find(lwci.GetId()) == m_InFlightOutputImages.end());

  std::map<unsigned int, LightweightCUDAImage>::iterator  i = m_ValidImages.find(lwci.GetId());
  assert(i != m_ValidImages.end());

  // this needs a check whether the readyevent for lwci was ever signaled!
  // otherwise we have a race condition with someone requesting an output image,
  // queueing a kernel, finalising, and immediately dropping the result.
  // that would trigger a call to here, but the queued kernel is still running so the image
  // is not available yet!
  if (!fromStreamCallback)
  {
    // as this function can also be called from a stream-callback (on which we are not allowed
    // to call any cuda api functions!) we need to guard for that case.
    // on stream-callback there is no need to sync on the ready-event: we know it's ready because
    // the callback happens after event signaling.
    cudaError_t err = cudaEventSynchronize(lwci.m_ReadyEvent);
    assert(err == cudaSuccess);
  }

  std::list<LightweightCUDAImage>&  freeList = m_AvailableImagePool[SizeToTier(lwci.m_SizeInBytes)];
  freeList.insert(freeList.begin(), lwci);

  // as the image is back on the free-list, it can no longer be read.
  // beware: m_ValidImages does not account for the refcount, so removing the image will recursively
  // call this method as its refcount goes to zero all the time.
  // work-around is to inc refcount. that works because in Finalise() we've dec'd the refcount specifically for m_ValidImages.
  lwci.m_RefCount->ref();
  m_ValidImages.erase(i);
}


//-----------------------------------------------------------------------------
void CUDART_CB CUDAManager::StreamCallback(cudaStream_t stream, cudaError_t status, void* userData)
{
  StreamCallbackReleasePOD*   pod = (StreamCallbackReleasePOD*) userData;

  pod->m_Manager->ReleaseReadAccess(pod->m_Id);

  // every callback is guaranteed to be executed only once. (says the cuda doc.)
  // so we are safe to free the memory.
  delete pod;
}


//-----------------------------------------------------------------------------
void CUDAManager::ReleaseReadAccess(unsigned int id)
{
  // FIXME: cannot grab a lock here! instead post a callback onto our thread.
  QMutexLocker    lock(&s_Lock);

  std::map<unsigned int, LightweightCUDAImage>::iterator i = m_ValidImages.find(id);
  assert(i != m_ValidImages.end());

  // this effectively drops the reference from the readaccessor (that is dead already) that was
  // passed into FinaliseAndAutorelease().
  bool dead = !i->second.m_RefCount->deref();
  if (dead)
  {
    AllRefsDropped(i->second, true);
  }
}
