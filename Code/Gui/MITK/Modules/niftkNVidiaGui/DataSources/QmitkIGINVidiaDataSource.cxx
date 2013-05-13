/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGINVidiaDataSource.h"
#include <mitkIGINVidiaDataType.h>
#include <../Conversion/ImageConversion.h>
#include <igtlTimeStamp.h>
#include <QTimer>
#include <QCoreApplication>
#include <QGLContext>
#include <QMutex>
#include <QGLWidget>
#include <QWaitCondition>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <video/sdiinput.h>
#include <video/compress.h>
#include <Mmsystem.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>


// FIXME: this needs tidying up
struct QmitkIGINVidiaDataSourceImpl
{
  // all the sdi stuff needs an opengl context
  //  so we'll create our own
  QGLWidget*              oglwin;
  // we want to share our capture context with other render contexts (e.g. the preview widget)
  // but for that to work we need a hack because for sharing to work, the share-source cannot
  //  be current at the time of call. but our capture context is current (to the capture thread)
  //  all the time! so we just create a dummy context that shares with capture-context but itself
  //  is never ever current to any thread and hence can be shared with new widgets while capture-context
  //  is happily working away. and tada it works :)
  QGLWidget*              oglshare;

  CUcontext               cuContext;

  enum CaptureState
  {
    PRE_INIT,
    HW_ENUM,
    FAILED,     // something is broken. signal dropout is not failed!
    RUNNING,    // trying to capture
    DEAD
  };
  // no need to lock this one
  volatile CaptureState   current_state;

  // any access to the capture bits needs to be locked
  mutable QMutex          lock;
  video::SDIDevice*       sdidev;
  video::SDIInput*        sdiin;
  video::StreamFormat     format;
  int                     streamcount;

  // we keep our own copy of the texture ids (instead of relying on sdiin)
  //  so that another thread can easily get these
  // SDIInput is actively enforcing an opengl context check that's incompatible
  //  with the current threading situation
  int                     textureids[4];

  video::Compressor*      compressor;

  volatile IplImage*      copyoutasap;
  QWaitCondition          copyoutfinished;
  QMutex                  copyoutmutex;
  int                     copyoutslot;      // which slot in the ringbuffer

  // maps sequence numbers to ringbuffer slots
  std::map<unsigned int, int>   sn2slot_map;
  // maps ringbuffer slots to sequence numbers
  std::map<int, unsigned int>   slot2sn_map;


  // time stamp of the previous successfully captured frame.
  // this is used to detect a capture glitch without unconditionally blocking for new frames.
  // see QmitkIGINVidiaDataSource::GrabData().
  DWORD     m_LastSuccessfulFrame;

  // used to detect whether record has stopped or not.
  // there's no notification when the user clicked stop-record.
  // QmitkIGINVidiaDataSource::GrabData(), bottom
  bool  m_WasSavingMessagesPreviously;

  // used in a log file to correlate times stamps, frame index and sequence number
  unsigned int    m_NumFramesCompressed;

public:
  QmitkIGINVidiaDataSourceImpl()
    : sdidev(0), sdiin(0), streamcount(0), oglwin(0), oglshare(0), cuContext(0), compressor(0), lock(QMutex::Recursive), 
      current_state(PRE_INIT), copyoutasap(0), m_LastSuccessfulFrame(0), m_WasSavingMessagesPreviously(false), m_NumFramesCompressed(0)
  {
    std::memset(&textureids[0], 0, sizeof(textureids));
    // we create the opengl widget on the ui thread once
    // and then never modify or signal/etc again
    oglwin = new QGLWidget(0, 0, Qt::WindowFlags(Qt::Window | Qt::FramelessWindowHint));
    oglwin->hide();
    assert(oglwin->isValid());

    // hack to get context sharing to work while the capture thread is cracking away
    oglshare = new QGLWidget(0, oglwin, Qt::WindowFlags(Qt::Window | Qt::FramelessWindowHint));
    oglshare->hide();
    assert(oglshare->isValid());
    assert(oglwin->isSharing());

    // we need to activate our capture context once for cuda setup
    QGLContext* prevctx = const_cast<QGLContext*>(QGLContext::currentContext());
    oglwin->makeCurrent();

    // Find out which gpu is rendering our window.
    // We need to know because this is where the video comes in
    // and on which we do compression.
    int           cudadevices[10];
    unsigned int  actualcudadevices = 0;
    // note that zero is a valid device index
    std::memset(&cudadevices[0], -1, sizeof(cudadevices));
    if (cudaGLGetDevices(&actualcudadevices, &cudadevices[0], sizeof(cudadevices) / sizeof(cudadevices[0]), cudaGLDeviceListAll) == cudaSuccess)
    {
      if (cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cudadevices[0]) == CUDA_SUCCESS)
      {
        cuCtxGetCurrent(&cuContext);
        // FIXME: do we need to pop the current context? is it leaking to the caller somehow?
        CUcontext oldctx;
        cuCtxPopCurrent(&oldctx);
      }
    }
    // else case is not very interesting: we could be running this on non-nvidia hardware
    // but we'll only know once we start enumerating sdi devices during QmitkIGINVidiaDataSource::GrabData()

    if (prevctx)
      prevctx->makeCurrent();
    else
      oglwin->doneCurrent();
  }

  ~QmitkIGINVidiaDataSourceImpl()
  {
    // we dont really support concurrent destruction
    // if some other thread is still holding on to a pointer
    //  while we are cleaning up here then things are going to blow up anyway
    // might as well fail fast
    bool wasnotlocked = lock.tryLock();
    assert(wasnotlocked);
    wasnotlocked = copyoutmutex.tryLock();
    assert(wasnotlocked);

    QGLContext* ctx = const_cast<QGLContext*>(QGLContext::currentContext());
    try
    {
      // we need the capture context for proper cleanup
      oglwin->makeCurrent();

      delete compressor;
      delete sdiin;
      // we do not own sdidev!
      sdidev = 0;
    
      if (ctx)
        ctx->makeCurrent();
      else
        oglwin->doneCurrent();
    }
    catch (...)
    {
        std::cerr << "sdi cleanup threw exception" << std::endl;
    }

    if (cuContext)
    {
      cuCtxDestroy(cuContext);
    }

    delete oglshare;
    delete oglwin;

    // they'll get cleaned up now
    // if someone else is currently waiting on these
    //  deleting but not unlocking does what to their waiting?
    copyoutmutex.unlock();
    lock.unlock();
  }

  // FIXME: needs cfg param to decide which channel to capture, format, etc
  void check_video(video::SDIInput::InterlacedBehaviour ib)
  {
    // make sure nobody messes around with contexts
    assert(QGLContext::currentContext() == oglwin->context());

    // we do not own the device!
    sdidev = 0;
    // but we gotta clear up this one
    delete sdiin;
    sdiin = 0;
    format = video::StreamFormat();
    // even though this pimpl class doesnt use the compressor directly
    // we should cleanup anyway.
    // so that whenever video dies we get a new file (because format could differ!)
    delete compressor;
    compressor = 0;

    // find our capture card
    for (int i = 0; ; ++i)
    {
      video::SDIDevice* d = video::SDIDevice::get_device(i);
      if (d == 0)
        break;

      if (d->get_type() == video::SDIDevice::INPUT)
      {
        sdidev = d;
        break;
      }
    }

    // so we have a card, check the incoming video format and hook up capture
    if (sdidev)
    {
      streamcount = 0;
      for (int i = 0; ; ++i, ++streamcount)
      {
        video::StreamFormat f = sdidev->get_format(i);
        if (f.format == video::StreamFormat::PF_NONE)
        {
          break;
        }

        format = f;
      }

      if (format.format != video::StreamFormat::PF_NONE)
      {
        sdiin = new video::SDIInput(sdidev, ib, format.get_refreshrate());
      }
    }

    // assuming everything went fine
    //  we now have texture objects that will receive video data everytime we call capture()
  }

  void readback_rgb(char* buffer, std::size_t bufferpitch, int width, int height)
  {
    assert(sdiin != 0);
    assert(bufferpitch >= width * 3);

    // lock here first, before querying dimensions so that there's no gap
    QMutexLocker    l(&lock);

    std::pair<int, int> dim = get_capture_dimensions();
    if (dim.first > width)
      // FIXME: should somehow communicate failure to the requester
      return;

    // unfortunately we have 3 bytes per pixel
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    assert((bufferpitch % 3) == 0);
    glPixelStorei(GL_PACK_ROW_LENGTH, bufferpitch / 3);

    for (int i = 0; i < 4; ++i)
    {
      int texid = sdiin->get_texture_id(i, copyoutslot);
      if (texid == 0)
        break;

      // while we have the lock, texture dimensions are not going to change

      // remaining buffer too small?
      if (height - (i * dim.second) < dim.second)
        return;

      char*   subbuf = &buffer[i * dim.second * bufferpitch];

      glBindTexture(GL_TEXTURE_2D, texid);
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, subbuf);
      assert(glGetError() == GL_NO_ERROR);
    }
  }
  void readback_rgba(char* buffer, std::size_t bufferpitch, int width, int height)
  {
    assert(sdiin != 0);
    assert(bufferpitch >= width * 4);

    // lock here first, before querying dimensions so that there's no gap
    QMutexLocker    l(&lock);

    std::pair<int, int> dim = get_capture_dimensions();
    if (dim.first > width)
      // FIXME: should somehow communicate failure to the requester
      return;

    // fortunately we have 4 bytes per pixel
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    assert((bufferpitch % 4) == 0);
    glPixelStorei(GL_PACK_ROW_LENGTH, bufferpitch / 4);

    for (int i = 0; i < 4; ++i)
    {
      int texid = sdiin->get_texture_id(i, copyoutslot);
      if (texid == 0)
        break;

      // while we have the lock, texture dimensions are not going to change

      // remaining buffer too small?
      if (height - (i * dim.second) < dim.second)
        return;

      char*   subbuf = &buffer[i * dim.second * bufferpitch];

      glBindTexture(GL_TEXTURE_2D, texid);
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, subbuf);
      assert(glGetError() == GL_NO_ERROR);
    }
  }

public:
  bool is_running() const
  {
    return current_state == RUNNING;
  }

  bool has_hardware() const
  {
    QMutexLocker    l(&lock);
    return sdidev != 0;
  }

  bool has_input() const
  {
    QMutexLocker    l(&lock);
    return sdiin != 0;
  }

  // this is the format reported by sdi
  // the actual capture format might be different!
  video::StreamFormat get_format() const
  {
    QMutexLocker    l(&lock);
    return format;
  }

  // might be different from advertised stream format
  std::pair<int, int> get_capture_dimensions() const
  {
    QMutexLocker    l(&lock);
    if (sdiin == 0)
        return std::make_pair(0, 0);
    return std::make_pair(sdiin->get_width(), sdiin->get_height());
  }

  int get_texture_id(unsigned int stream) const
  {
    QMutexLocker    l(&lock);
    if (sdiin == 0)
        return 0;
    if (stream >= 4)
        return 0;
    return textureids[stream];
  }

  int get_stream_count() const
  {
    QMutexLocker    l(&lock);
    return streamcount;
  }
};


// note the trailing space
const char* QmitkIGINVidiaDataSource::s_NODE_NAME = "NVIDIA SDI stream ";


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource::QmitkIGINVidiaDataSource(mitk::DataStorage* storage)
: QmitkIGILocalDataSource(storage)
, m_Pimpl(new QmitkIGINVidiaDataSourceImpl), m_MipmapLevel(0)
// NOTE: has to match GUI's default value!
, m_FieldMode(STACK_FIELDS)
{
  this->SetName("QmitkIGINVidiaDataSource");
  this->SetType("Frame Grabber");
  this->SetDescription("NVidia SDI");
  this->SetStatus("Initialising...");

  // pre-create any number of datastorage nodes to avoid threading issues
  for (int i = 0; i < 4; ++i)
  {
    std::ostringstream  nodename;
    nodename << s_NODE_NAME << i;

    mitk::DataNode::Pointer node = this->GetDataNode(nodename.str());
  }

  StartCapturing();
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::SetMipmapLevel(unsigned int l)
{
  if (IsCapturing())
  {
    throw std::runtime_error("Cannot change capture parameter while capture is in progress");
  }

  // whether the requested level is meaningful or not depends on the incoming video size.
  // i dont want to check that here!
  // so we just silently accept most values and clamp it later.

  if (l > 32)
  {
    // this would mean we have an input video of more than 4 billion by 4 billion pixels.
    // while technology will eventually get there, i think it's safe to assume that
    // somebody would have checked this code here by then.
    throw std::runtime_error("Requested mipmap level is not sane (> 32)");
  }

  m_MipmapLevel = l;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::SetFieldMode(InterlacedBehaviour b)
{
  if (IsCapturing())
  {
    throw std::runtime_error("Cannot change capture parameter while capture is in progress");
  }

  m_FieldMode = b;
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource::~QmitkIGINVidiaDataSource()
{
  // Try stop grabbing and threading etc.
  // We do need quite a bit of control over the actual threading setup because
  // we need to manage which thread is currently in charge of the capture context!
  this->StopCapturing();

  delete m_Pimpl;
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::CanHandleData(mitk::IGIDataType* data) const
{
  bool result = false;
  if (static_cast<mitk::IGINVidiaDataType*>(data) != NULL)
  {
    result = true;
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::StartCapturing()
{
  // we need to reset this here before the grabbing thread starts
  // because it will check whether it has the correct opengl context.
  // and that check would fail if we stop() and then restart capturing.
  m_Pimpl->current_state = QmitkIGINVidiaDataSourceImpl::PRE_INIT;
  this->InitializeAndRunGrabbingThread(20);
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::StopCapturing()
{
  StopGrabbingThread();
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::IsCapturing()
{
  bool result = m_GrabbingThread != 0;
  return result;
}

/*
std::pair<IplImage*, int> QmitkIGINVidiaDataSource::GetRgbImage()
{
  // i dont like this way of unstructured locking
  m_Pimpl->lock.lock();
  // but the waitcondition stuff doesnt work otherwise

  std::pair<int, int>   imgdim = m_Pimpl->get_capture_dimensions();
  int                   streamcount = m_Pimpl->get_stream_count();

  IplImage* frame = cvCreateImage(cvSize(imgdim.first, imgdim.second * streamcount), IPL_DEPTH_8U, 3);
  // mark layout as rgb instead of the opencv-default bgr
  std::memcpy(&frame->channelSeq[0], "RGB\0", 4);

  m_Pimpl->copyoutasap = frame;

  // this smells like deadlock...
  m_Pimpl->copyoutmutex.lock();
  // until here, capture thread would be stuck waiting for the lock
  m_Pimpl->lock.unlock();

  // FIXME: we should bump m_GrabbingThread so it wakes up early from its message loop sleep
  //        otherwise we are locking in on its refresh rate

  m_Pimpl->copyoutfinished.wait(&m_Pimpl->copyoutmutex);
  m_Pimpl->copyoutmutex.unlock();
  return std::make_pair(frame, streamcount);
}
*/

//-----------------------------------------------------------------------------
std::pair<IplImage*, int> QmitkIGINVidiaDataSource::GetRgbaImage(unsigned int sequencenumber)
{
  // i dont like this way of unstructured locking
  m_Pimpl->lock.lock();
  // but the waitcondition stuff doesnt work otherwise

  // check if we ever have received any frames yet
  if (m_Pimpl->sn2slot_map.empty())
  {
    m_Pimpl->lock.unlock();
    return std::make_pair((IplImage*) 0, 0);
  }

  std::pair<int, int>   imgdim = m_Pimpl->get_capture_dimensions();
  int                   streamcount = m_Pimpl->get_stream_count();

  IplImage* frame = cvCreateImage(cvSize(imgdim.first, imgdim.second * streamcount), IPL_DEPTH_8U, 4);
  // mark layout as rgba instead of the opencv-default bgr
  std::memcpy(&frame->channelSeq[0], "RGBA", 4);

  m_Pimpl->copyoutasap = frame;
  m_Pimpl->copyoutslot = m_Pimpl->sn2slot_map.lower_bound(sequencenumber)->second;

  // this smells like deadlock...
  m_Pimpl->copyoutmutex.lock();
  // until here, capture thread would be stuck waiting for the lock
  m_Pimpl->lock.unlock();

  // FIXME: we should bump m_GrabbingThread so it wakes up early from its message loop sleep
  //        otherwise we are locking in on its refresh rate

  m_Pimpl->copyoutfinished.wait(&m_Pimpl->copyoutmutex);
  m_Pimpl->copyoutmutex.unlock();
  return std::make_pair(frame, streamcount);
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::GrabData()
{
  assert(m_Pimpl != 0);
  assert(m_GrabbingThread == (QmitkIGILocalDataSourceGrabbingThread*) QThread::currentThread());


  QMutexLocker    l(&m_Pimpl->lock);

  if (m_Pimpl->current_state == QmitkIGINVidiaDataSourceImpl::FAILED)
  {
    return;
  }
  if (m_Pimpl->current_state == QmitkIGINVidiaDataSourceImpl::DEAD)
  {
    return;
  }

  if (m_Pimpl->current_state == QmitkIGINVidiaDataSourceImpl::PRE_INIT)
  {
    m_Pimpl->oglwin->makeCurrent();
    m_Pimpl->current_state = QmitkIGINVidiaDataSourceImpl::HW_ENUM;
  }

  // make sure nobody messes around with contexts
  assert(QGLContext::currentContext() == m_Pimpl->oglwin->context());

  if (m_Pimpl->current_state == QmitkIGINVidiaDataSourceImpl::HW_ENUM)
  {
    try
    {
      // libvideo does its own glew init, so we can get cracking straight away
      m_Pimpl->check_video((video::SDIInput::InterlacedBehaviour) m_FieldMode);

      // once we have an input setup
      //  grab at least one frame (not quite sure why)
      if (m_Pimpl->sdiin)
      {
        m_Pimpl->sdiin->capture();
      }
    }
    // getting an exception means something is broken
    // during normal operation this should never happen
    //  even if there's no hardware or signal
    catch (const std::exception& e)
    {
      this->SetStatus(std::string("Failed: ") + e.what());
      m_Pimpl->current_state = QmitkIGINVidiaDataSourceImpl::FAILED;
      return;
    }
    catch (...)
    {
      this->SetStatus("Failed");
      m_Pimpl->current_state = QmitkIGINVidiaDataSourceImpl::FAILED;
      return;
    }
  }

  if (!m_Pimpl->has_hardware())
  {
    this->SetStatus("No SDI hardware");
    // no hardware then nothing to do
    m_Pimpl->current_state = QmitkIGINVidiaDataSourceImpl::DEAD;
    return;
  }

  if (!m_Pimpl->has_input())
  {
    this->SetStatus("No input signal");
    // no signal, try again next round
    m_Pimpl->current_state = QmitkIGINVidiaDataSourceImpl::HW_ENUM;
    return;
  }

  // if we get to here then we should be good to go!
  m_Pimpl->current_state = QmitkIGINVidiaDataSourceImpl::RUNNING;

  try
  {
    bool hasframe = m_Pimpl->sdiin->has_frame();
    // note: has_frame() will not throw an exception in case setup is broken

    // make sure we try to capture a frame if the previous one was too long ago.
    // that will check for errors and throw an exception if necessary, which will then allow us to restart.
    if ((timeGetTime() - m_Pimpl->m_LastSuccessfulFrame) > 1000)
      hasframe = true;


    if (hasframe)
    {
      // note: capture() will block for a frame to arrive
      // that's why we have hasframe above
      video::FrameInfo fi = m_Pimpl->sdiin->capture();
      m_Pimpl->m_LastSuccessfulFrame = timeGetTime();

      // keep the most recent set of texture ids around
      // this is mainly for the preview window
      for (int i = 0; i < 4; ++i)
        m_Pimpl->textureids[i] = m_Pimpl->sdiin->get_texture_id(i, -1);

      int newest_slot = m_Pimpl->sdiin->get_current_ringbuffer_slot();
      // whatever we had in this slot is now obsolete
      std::map<int, unsigned int>::iterator oldsni = m_Pimpl->slot2sn_map.find(newest_slot);
      if (oldsni != m_Pimpl->slot2sn_map.end())
      {
        std::map<unsigned int, int>::iterator oldsloti = m_Pimpl->sn2slot_map.find(oldsni->second);
        if (oldsloti != m_Pimpl->sn2slot_map.end())
        {
          m_Pimpl->sn2slot_map.erase(oldsloti);
        }
        m_Pimpl->slot2sn_map.erase(oldsni);
      }
      m_Pimpl->slot2sn_map[newest_slot] = fi.sequence_number;
      m_Pimpl->sn2slot_map[fi.sequence_number] = newest_slot;

      igtl::TimeStamp::Pointer timeCreated = igtl::TimeStamp::New();

      mitk::IGINVidiaDataType::Pointer wrapper = mitk::IGINVidiaDataType::New();
      wrapper->SetValues(1, fi.sequence_number, fi.arrival_time);
      wrapper->SetDataSource("QmitkIGINVidiaDataSource");
      wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(timeCreated));
      wrapper->SetDuration(1000000000); // nanoseconds

      // under certain circumstances, this might call back into SaveData()
      this->AddData(wrapper.GetPointer());

      this->SetStatus("Grabbing");
    }
  }
  // capture() might throw if the capture setup has become invalid
  // e.g. a mode change or signal lost
  catch (...)
  {
    this->SetStatus("Glitched out");
    m_Pimpl->current_state = QmitkIGINVidiaDataSourceImpl::HW_ENUM;
    return;
  }

  // dont put the copy-out bits in the has_frame condition
  // otherwise we are again locking the datastorage-update-thread onto the sdi refresh rate
  if (m_Pimpl->copyoutasap)
  {
    if (m_Pimpl->copyoutasap->nChannels == 3)
    {
      m_Pimpl->readback_rgb(m_Pimpl->copyoutasap->imageData, m_Pimpl->copyoutasap->widthStep, m_Pimpl->copyoutasap->width, m_Pimpl->copyoutasap->height);
    }
    else
    if (m_Pimpl->copyoutasap->nChannels == 4)
    {
      m_Pimpl->readback_rgba(m_Pimpl->copyoutasap->imageData, m_Pimpl->copyoutasap->widthStep, m_Pimpl->copyoutasap->width, m_Pimpl->copyoutasap->height);
    }
    else
    {
      assert(false);
    }
    m_Pimpl->copyoutasap = 0;
    m_Pimpl->copyoutfinished.wakeOne();
  }

  // because there's currently no notification when the user clicked stop-record
  // we need to check this way to clean up the compressor.
  if (m_Pimpl->m_WasSavingMessagesPreviously && !GetSavingMessages())
  {
    delete m_Pimpl->compressor;
    m_Pimpl->compressor = 0;
    m_Pimpl->m_WasSavingMessagesPreviously = false;

    m_FrameMapLogFile.close();
  }
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::Update(mitk::IGIDataType* data)
{
  bool result = true;

  mitk::IGINVidiaDataType::Pointer dataType = static_cast<mitk::IGINVidiaDataType*>(data);
  if (dataType.IsNotNull())
  {
    // one massive image, with all streams stacked in
    std::pair<IplImage*, int> frame = GetRgbaImage(dataType->GetSequenceNumber());
    // if copy-out failed then capture setup is broken, e.g. someone unplugged a cable
    if (frame.first)
    {
      // max 4 streams
      const int streamcount = frame.second;
      for (int i = 0; i < streamcount; ++i)
      {
        std::ostringstream  nodename;
        nodename << s_NODE_NAME << i;

        mitk::DataNode::Pointer node = this->GetDataNode(nodename.str());
        if (node.IsNull())
        {
          MITK_ERROR << "Can't find mitk::DataNode with name " << nodename.str() << std::endl;
          this->SetStatus("Failed");
          cvReleaseImage(&frame.first);
          return false;
        }

        // subimagheight counts for both fields, stacked on top of each other...
        int   subimagheight = frame.first->height / streamcount;
        IplImage  subimg;
        // ...while subimg will have half the height only, i.e. the top field only
        cvInitImageHeader(&subimg, cvSize((int) frame.first->width, subimagheight / 2), IPL_DEPTH_8U, frame.first->nChannels);
        cvSetData(&subimg, &frame.first->imageData[i * subimagheight * frame.first->widthStep], frame.first->widthStep);

        // Check if we already have an image on the node.
        // We dont want to create a new one in that case (there is a lot of stuff going on
        // for allocating a new image).
        mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());

        if (!imageInNode.IsNull())
        {
          // check size of image that is already attached to data node!
          bool haswrongsize = false;
          haswrongsize |= imageInNode->GetDimension(0) != subimg.width;
          haswrongsize |= imageInNode->GetDimension(1) != subimg.height;
          haswrongsize |= imageInNode->GetDimension(2) != 1;

          if (haswrongsize)
          {
            imageInNode = mitk::Image::Pointer();
          }
        }

        if (imageInNode.IsNull())
        {
          mitk::Image::Pointer convertedImage = niftk::CreateMitkImage(&subimg);
          // our image is only half the pixel height!
          // so tell the renderer/mitk/whatever that each pixel is actually two units tall
          mitk::Vector3D  s = convertedImage->GetGeometry()->GetSpacing();
          s[1] *= 2;
          convertedImage->GetGeometry()->SetSpacing(s);

          node->SetData(convertedImage);
        }
        else
        {
          try
          {
            mitk::ImageWriteAccessor writeAccess(imageInNode);
            void* vPointer = writeAccess.GetData();

            // the mitk image is tightly packed
            // but the opencv image might not
            const unsigned int numberOfBytesPerLine = subimg.width * subimg.nChannels;
            if (numberOfBytesPerLine == static_cast<unsigned int>(subimg.widthStep))
            {
              std::memcpy(vPointer, subimg.imageData, numberOfBytesPerLine * subimg.height);
            }
            else
            {
              // if that is not true then something is seriously borked
              assert(subimg.widthStep >= numberOfBytesPerLine);

              // "slow" path: copy line by line
              for (int y = 0; y < subimg.height; ++y)
              {
                // widthStep is in bytes while width is in pixels
                std::memcpy(&(((char*) vPointer)[y * numberOfBytesPerLine]), &(subimg.imageData[y * subimg.widthStep]), numberOfBytesPerLine); 
              }
            }
          }
          catch(mitk::Exception& e)
          {
            MITK_ERROR << "Failed to copy OpenCV image to DataStorage due to " << e.what() << std::endl;
          }
        }
        node->Modified();
      } // for

      cvReleaseImage(&frame.first);
    } 
    else
      this->SetStatus("Failed");
  }
  else
    this->SetStatus("...");

  // We signal every time we receive data, rather than at the GUI refresh rate, otherwise video looks very odd.
  emit UpdateDisplay();

  return result;
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::SaveData(mitk::IGIDataType* data, std::string& outputFileName)
{
  assert(m_Pimpl != 0);
  assert(m_GrabbingThread == (QmitkIGILocalDataSourceGrabbingThread*) QThread::currentThread());

  bool success = false;
  outputFileName = "";

  mitk::IGINVidiaDataType::Pointer dataType = static_cast<mitk::IGINVidiaDataType*>(data);
  if (dataType.IsNotNull())
  {
    QMutexLocker    l(&m_Pimpl->lock);

    // make sure nobody messes around with contexts
    assert(QGLContext::currentContext() == m_Pimpl->oglwin->context());

    // no record-stop-notification work around
    m_Pimpl->m_WasSavingMessagesPreviously = true;

    if (m_Pimpl->compressor == 0)
    {
      CUresult r = cuCtxPushCurrent(m_Pimpl->cuContext);
      // die straight away
      if (r != CUDA_SUCCESS)
        return false;

      std::pair<int, int> dim = m_Pimpl->get_capture_dimensions();

      // FIXME: use qt for this
      SYSTEMTIME  now;
      GetSystemTime(&now);

      std::string directoryPath = GetSavePrefix() + '/' + "QmitkIGINVidiaDataSource";
      QDir directory(QString::fromStdString(directoryPath));
      if (directory.mkpath(QString::fromStdString(directoryPath)))
      {
        std::ostringstream    filename;
        filename << directoryPath << "/capture-" 
          << now.wYear << '_' << now.wMonth << '_' << now.wDay << '-' << now.wHour << '_' << now.wMinute << '_' << now.wSecond;

        std::string filenamebase = filename.str();

        // we need a map for frame number -> wall clock
        assert(!m_FrameMapLogFile.is_open());
        m_FrameMapLogFile.open((filenamebase + ".framemap.log").c_str());
        if (!m_FrameMapLogFile.is_open())
        {
          // should we continue if we dont have a frame map?
          std::cerr << "WARNING: could not create frame map file!" << std::endl;
        }
        else
        {
          // dump a header line
          m_FrameMapLogFile << "#framenumber sequencenumber channel timestamp" << std::endl;
        }

        // also keep sdi logs
        m_Pimpl->sdiin->set_log_filename(filenamebase + ".sdicapture.log");

        // when we get a new compressor we want to start counting from zero again
        m_Pimpl->m_NumFramesCompressed = 0;

        m_Pimpl->compressor = new video::Compressor(dim.first, dim.second, m_Pimpl->format.refreshrate * m_Pimpl->streamcount, filenamebase + ".264");
      }
    }
    else
    {
      // we have compressor already so context should be all set up
      // check it!
      CUcontext ctx = 0;
      CUresult r = cuCtxGetCurrent(&ctx);
      // if for any reason we cant interact with cuda then there's no use of trying to do anything else
      if (r != CUDA_SUCCESS)
        return false;
      assert(ctx == m_Pimpl->cuContext);
    }

    // find out which ringbuffer slot the request sequence number is in, if any
    unsigned int requestedSN = dataType->GetSequenceNumber();
    std::map<unsigned int, int>::iterator sloti = m_Pimpl->sn2slot_map.find(requestedSN);
    if (sloti != m_Pimpl->sn2slot_map.end())
    {
      // sanity check
      assert(m_Pimpl->slot2sn_map.find(sloti->second)->second == requestedSN);

      // compress each stream
      for (int i = 0; i < m_Pimpl->streamcount; ++i)
      {
        int tid = m_Pimpl->sdiin->get_texture_id(i, sloti->second);
        assert(tid != 0);
        // would need to do prepare() only once
        // but more often is ok too
        m_Pimpl->compressor->preparetexture(tid);
        m_Pimpl->compressor->compresstexture(tid);

        if (m_FrameMapLogFile.is_open())
        {
          m_FrameMapLogFile 
            << m_Pimpl->m_NumFramesCompressed << '\t' 
            << requestedSN << '\t'
            << i << '\t'
            << dataType->GetTimeStampInNanoSeconds() << '\t'
            << std::endl;
        }

        m_Pimpl->m_NumFramesCompressed++;
      }

      success = true;
    }
    else
    {
      assert(false);
    }
  }

  return success;
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSource::GetNumberOfStreams()
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  return m_Pimpl->get_stream_count();
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSource::GetCaptureWidth()
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  video::StreamFormat format = m_Pimpl->get_format();
  return format.get_width();
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSource::GetCaptureHeight()
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  video::StreamFormat format = m_Pimpl->get_format();
  return format.get_height();
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSource::GetRefreshRate()
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  video::StreamFormat format = m_Pimpl->get_format();
  return format.get_refreshrate();
}


//-----------------------------------------------------------------------------
QGLWidget* QmitkIGINVidiaDataSource::GetCaptureContext()
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  assert(m_Pimpl->oglshare != 0);
  return m_Pimpl->oglshare;
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSource::GetTextureId(int stream)
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  return m_Pimpl->get_texture_id(stream);
}
