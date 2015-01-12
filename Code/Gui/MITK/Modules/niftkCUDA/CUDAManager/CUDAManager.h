/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef CUDAManager_h
#define CUDAManager_h

#include "niftkCUDAExports.h"
#include <CUDAImage/CUDAImage.h>
#include <CUDAImage/LightweightCUDAImage.h>
#include <QThread>
#include <QMutex>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <list>
#include <set>
#include <string>
#include <boost/lockfree/queue.hpp>


// FIXME: not yet implemented
struct ScopedCUDADevice
{
  ScopedCUDADevice(int dev);
  ~ScopedCUDADevice();
};


/**
 * Holds information necessary to read from a LightweightCUDAImage.
 * @see CUDAManager::RequestReadAccess
 */
struct ReadAccessor
{
  const void*     m_DevicePointer;
  std::size_t     m_SizeInBytes;
  unsigned int    m_BytePitch;
  unsigned int    m_PixelWidth;
  unsigned int    m_PixelHeight;            // obviously the unit is lines of pixels
  int             m_FIXME_pixeltype;        // still havent thought about this one...

  unsigned int    m_Id;
  cudaEvent_t     m_ReadyEvent;
};


/**
 * Holds information necessary to write into a CUDA memory block
 * representing an image.
 * @see CUDAManager::RequestOutputImage
 */
struct WriteAccessor
{
  void*           m_DevicePointer;
  std::size_t     m_SizeInBytes;
  unsigned int    m_BytePitch;
  unsigned int    m_PixelWidth;
  unsigned int    m_PixelHeight;            // obviously the unit is lines of pixels
  int             m_FIXME_pixeltype;        // still havent thought about this one...

  unsigned int    m_Id;
  cudaEvent_t     m_ReadyEvent;
};


// forward-decl
namespace impldetail
{
struct ModuleCleanup;
struct StreamCallbackReleasePOD;
}


/**
 * Singleton that owns all CUDA resources.
 * It manages images in a copy-on-write like fashion: you cannot write into an existing
 * CUDA-image, you can only read from these and write data into a newly allocated one.
 *
 * To get access to an image living on the card, do the usual DataNode::GetData(), and a
 * cast to CUDAImage. Then call CUDAImage::GetLightweightCUDAImage() to retrieve a handle
 * to the actual bits in CUDA-memory.
 * Side note: even though LightweightCUDAImage has members you should consider it opaque.
 *
 * This LightweightCUDAImage instance you can use with RequestReadAccess() to obtain a device
 * pointer that you can read from in your kernel.
 * RequestReadAccess() will increment a reference count for that image so that CUDAManager
 * will not recycle it too early.
 *
 * Then call RequestOutputImage() to get a device pointer to where you can write your kernel's
 * output. From an API point of view, RequestOutputImage() will always give you a new memory
 * block so that you never overwrite an existing image.
 *
 * Call GetStream() with your favourite name, or create your own stream, for synchronising and
 * coarse-grain parallelising CUDA tasks.
 *
 * Run your kernel on your stream. But do not synchronise on its completion!
 *
 * When all your work has been submitted to the driver, call FinaliseAndAutorelease() to turn
 * the output device pointer into a proper LightweightCUDAImage that you can stick onto a
 * DataNode. This function will also release your read-request on the input image at the right time
 * so that it can be eventually returned to the memory pool.
 * In addition, Finalise*() functions will queue a "ready" event that you can use on your stream to
 * GPU-synchronise on completion of a previous processing step.
 *
 * CUDAManager is thread-safe: all public methods can be called from any thread at any time.
 */
class NIFTKCUDA_EXPORT CUDAManager : public QThread
{
  friend class LightweightCUDAImage;
  friend struct impldetail::ModuleCleanup;

public:
  /**
   *
   * @throws std::runtime_error if CUDA is not available on the system.
   */
  static CUDAManager* GetInstance();


  // FIXME: not yet implemented
  ScopedCUDADevice ActivateDevice(int dev);

  cudaStream_t GetStream(const std::string& name);

  /**
   * @throws std::runtime_error if lwci is not valid.
   */
  ReadAccessor RequestReadAccess(const LightweightCUDAImage& lwci);

  WriteAccessor RequestOutputImage(unsigned int width, unsigned int height, int FIXME_pixeltype);

  // when done with queueing commands to fill output image, call this.
  // it will give you a LightweightCUDAImage that can be stuffed in CUDAImage,
  // which in turn can go to a DataNode.
  LightweightCUDAImage Finalise(WriteAccessor& writeAccessor, cudaStream_t stream);

  /**
   * Combines Finalise() and Autorelease() into a single call.
   */
  LightweightCUDAImage FinaliseAndAutorelease(WriteAccessor& writeAccessor, ReadAccessor& readAccessor, cudaStream_t stream);

  /**
   * Releases the read-request of an image once processing on stream has finished.
   * This method does not block, it will return immediately.
   * Make sure you call this method after Finalise(), or use FinaliseAndAutorelease().
   */
  void Autorelease(ReadAccessor& readAccessor, cudaStream_t stream);


  void Autorelease(WriteAccessor& writeAccessor, cudaStream_t stream);


protected:
  CUDAManager();
  virtual ~CUDAManager();


  /**
   * Used by LightweightCUDAImage to notify us that all references to it have been dropped,
   * and that it can be placed back onto m_AvailableImagePool for later re-use.
   */
  void AllRefsDropped(LightweightCUDAImage& lwci);


  /** Copy and assignment is not allowed. */
  //@{
private:
  CUDAManager(const CUDAManager& copyme);
  CUDAManager& operator=(const CUDAManager& assignme);
  //@}


  /** Convert a size-tier into a maximum byte size that would fit into that tier. */
  std::size_t TierToSize(unsigned int tier) const;

  /** Convert a byte size into a tier that it falls into. */
  unsigned int SizeToTier(std::size_t size) const;


  /**
   * Called by the CUDA driver when work has finished.
   * The callback is queued by FinaliseAndAutorelease() to release an image.
   * Note: this callback will block work on the stream, therefore the image-ready-events are
   * triggered before the callback so that work on other streams can proceed in parallel.
   * @internal
   */
  static void CUDART_CB AutoReleaseStreamCallback(cudaStream_t stream, cudaError_t status, void* userData);


  /**
   * Called by StreamCallback (which in turn is triggered by FinaliseAndAutorelease()) to "release"
   * a previously requested image.
   * @internal
   */
  void ReleaseReadAccess(unsigned int id);

  /**
   * Called at opportune moments to free up items on m_AutoreleaseQueue.
   * @internal
   */
  void ProcessAutoreleaseQueue();


  static CUDAManager*           s_Instance;
  // there's only one instance of our class (singleton), so a single mutex is ok too.
  static QMutex                 s_Lock;

  unsigned int                  m_LastIssuedId;

  // vector is a size tier, followed by linked list for that tier.
  std::vector<std::list<LightweightCUDAImage> >     m_AvailableImagePool;

  // images currently in use via WriteAccessor, i.e. work is being queued.
  std::map<unsigned int, LightweightCUDAImage>    m_InFlightOutputImages;

  // images that can be requested with RequestReadAccess.
  std::map<unsigned int, LightweightCUDAImage>    m_ValidImages;

  std::map<std::string, cudaStream_t>     m_Streams;

  // the auto-release callback cannot acquire s_Lock because that will deadlock within the cuda driver.
  boost::lockfree::queue<impldetail::StreamCallbackReleasePOD*>     m_AutoreleaseQueue;
};


#endif // CUDAManager_h
