/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGINVidiaDataSourceImpl.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <Mmsystem.h>
#include <opencv2/core/core_c.h>
#include <igtlTimeStamp.h>
#include <boost/typeof/typeof.hpp>
#include <NiftyLinkUtils.h>
#include <fstream>


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
QmitkIGINVidiaDataSourceImpl::QmitkIGINVidiaDataSourceImpl()
  : QmitkIGITimerBasedThread(0),
    sdidev(0), sdiin(0), streamcount(0), oglwin(0), oglshare(0), cuContext(0), compressor(0), decompressor(0),
    lock(QMutex::Recursive), 
    current_state(PRE_INIT), m_LastSuccessfulFrame(0), m_NumFramesCompressed(0), wireformat(""),
    m_Cookie(0),
    state_message("Starting up")
{
  // check whether cuda dlls are available
  if (!CUDADelayLoadCheck())
  {
    throw std::runtime_error("No suitable CUDA available");
  }

  // helps with debugging
  setObjectName("QmitkIGINVidiaDataSourceImpl");
  // sample interval in milliseconds that a timer will kick in to check for sdi hardware.
  // once capture has been setup, it will trigger much more often to keep up with video refresh rate.
  SetInterval(100);


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


  // we want signal/slot processing to happen on our background thread.
  // for that to work we need to explicitly move this object because
  // it is currently owned by the gui thread.
  // FIXME: i wonder how well that works with start() and quit(). our thread instance stays
  //        the same so it should be ok?
  this->moveToThread(this);
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSourceImpl::~QmitkIGINVidiaDataSourceImpl()
{
  // we dont really support concurrent destruction
  // if some other thread is still holding on to a pointer
  //  while we are cleaning up here then things are going to blow up anyway
  // might as well fail fast
  bool wasnotlocked = lock.tryLock();
  assert(wasnotlocked);

  QGLContext* ctx = const_cast<QGLContext*>(QGLContext::currentContext());
  try
  {
    // we need the capture context for proper cleanup
    oglwin->makeCurrent();

    delete decompressor;
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
  lock.unlock();
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::InitVideo()
{
  // make sure nobody messes around with contexts
  assert(QGLContext::currentContext() == oglwin->context());

  // we do not own the device!
  sdidev = 0;
  // but we gotta clear up this one
  delete sdiin;
  sdiin = 0;
  format = video::StreamFormat();
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
      wireformat = sdidev->get_wireformat();
      if (f.format == video::StreamFormat::PF_NONE)
      {
        break;
      }

      format = f;
    }

    if (format.format != video::StreamFormat::PF_NONE)
    {
      sdiin = new video::SDIInput(sdidev, m_FieldMode, format.get_refreshrate());

      m_Cookie = (unsigned int) ((((std::size_t) ((void*) sdiin)) >> 4) & 0xFFFFFFFF);
    }
  }

  // assuming everything went fine
  //  we now have texture objects that will receive video data everytime we call capture()
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::ReadbackRGBA(char* buffer, std::size_t bufferpitch, int width, int height, int slot)
{
  assert(sdiin != 0);
  assert(bufferpitch >= width * 4);

  int w = sdiin->get_width();
  int h = sdiin->get_height();

  assert(w <= width);


  // fortunately we have 4 bytes per pixel
  glPixelStorei(GL_PACK_ALIGNMENT, 4);
  assert((bufferpitch % 4) == 0);
  glPixelStorei(GL_PACK_ROW_LENGTH, bufferpitch / 4);

  for (int i = 0; i < 4; ++i)
  {
    int texid = sdiin->get_texture_id(i, slot);
    if (texid == 0)
      break;

    // while we have the lock, texture dimensions are not going to change

    // remaining buffer too small?
    if (height - (i * h) < h)
      return;

    char*   subbuf = &buffer[i * h * bufferpitch];

    glBindTexture(GL_TEXTURE_2D, texid);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, subbuf);
    assert(glGetError() == GL_NO_ERROR);
  }
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSourceImpl::IsRunning() const
{
  QMutexLocker    l(&lock);
  return current_state == RUNNING;
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSourceImpl::HasHardware() const
{
  QMutexLocker    l(&lock);
  return sdidev != 0;
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSourceImpl::HasInput() const
{
  QMutexLocker    l(&lock);
  return sdiin != 0;
}


//-----------------------------------------------------------------------------
unsigned int QmitkIGINVidiaDataSourceImpl::GetCookie() const
{
  QMutexLocker    l(&lock);
  return m_Cookie;
}


//-----------------------------------------------------------------------------
const char* QmitkIGINVidiaDataSourceImpl::GetWireFormatString() const
{
  QMutexLocker    l(&lock);
  return wireformat;
}


//-----------------------------------------------------------------------------
video::FrameInfo QmitkIGINVidiaDataSourceImpl::GetNextSequenceNumber(unsigned int ihavealready) const
{
  QMutexLocker    l(&lock);

  video::FrameInfo  fi = {0};
  fi.sequence_number = ihavealready;

  BOOST_TYPEOF(sn2slot_map)::const_iterator i = sn2slot_map.upper_bound(fi);
  if (i == sn2slot_map.end())
  {
    video::FrameInfo  empty = {0};
    return empty;
  }

  return i->first;
}


//-----------------------------------------------------------------------------
// this is the format reported by sdi
// the actual capture format might be different!
video::StreamFormat QmitkIGINVidiaDataSourceImpl::GetFormat() const
{
  QMutexLocker    l(&lock);
  return format;
}


//-----------------------------------------------------------------------------
QGLWidget* QmitkIGINVidiaDataSourceImpl::GetCaptureContext()
{
  QMutexLocker    l(&lock);
  assert(oglshare != 0);
  return oglshare;
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSourceImpl::GetTextureId(unsigned int stream) const
{
  QMutexLocker    l(&lock);
  if (sdiin == 0)
      return 0;
  if (stream >= 4)
      return 0;
  return textureids[stream];
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSourceImpl::GetStreamCount() const
{
  QMutexLocker    l(&lock);
  return streamcount;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::SetFieldMode(video::SDIInput::InterlacedBehaviour mode)
{
  QMutexLocker    l(&lock);
  m_FieldMode = mode;
}


//-----------------------------------------------------------------------------
video::SDIInput::InterlacedBehaviour QmitkIGINVidiaDataSourceImpl::GetFieldMode() const
{
  QMutexLocker    l(&lock);
  return m_FieldMode;
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSourceImpl::CaptureState QmitkIGINVidiaDataSourceImpl::GetCaptureState() const
{
  QMutexLocker    l(&lock);
  return current_state;
}


//-----------------------------------------------------------------------------
std::string QmitkIGINVidiaDataSourceImpl::GetStateMessage() const
{
  QMutexLocker    l(&lock);
  return state_message;
}


//-----------------------------------------------------------------------------
std::string QmitkIGINVidiaDataSourceImpl::GetCompressionOutputFilename() const
{
  QMutexLocker    l(&lock);
  return m_CompressionOutputFilename;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::setCompressionOutputFilename(const std::string& name)
{
  QMutexLocker    l(&lock);
  m_CompressionOutputFilename = name;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::run()
{
  // make sure the correct opengl context is active
  // OnTimeoutImpl() there's another make-current, but there were circumstances in which that was too late.
  oglwin->makeCurrent();

  // it's possible for someone else to start and stop our thread.
  // just make sure we start clean if that happens.
  Reset();

  bool ok = connect(this, SIGNAL(SignalBump()), this, SLOT(DoWakeUp()), Qt::QueuedConnection);
  assert(ok);
  ok = connect(this, SIGNAL(SignalCompress(unsigned int, unsigned int*)), this, SLOT(DoCompressFrame(unsigned int, unsigned int*)), Qt::BlockingQueuedConnection);
  assert(ok);
  ok = connect(this, SIGNAL(SignalStopCompression()), this, SLOT(DoStopCompression()), Qt::BlockingQueuedConnection);
  assert(ok);
  ok = connect(this, SIGNAL(SignalGetRGBAImage(unsigned int, IplImage**, unsigned int*)), this, SLOT(DoGetRGBAImage(unsigned int, IplImage**, unsigned int*)), Qt::BlockingQueuedConnection);
  assert(ok);

  // current_state is reset by Reset() called above.
  // but is this a requirement for our message loop to run properly?
  // i think it is: whenever somebody (re-)starts our sdi thread we can assume
  // that our previous capture-setup is now dead/stale.
  assert(current_state == PRE_INIT);

  // let base class deal with timer and event loop and stuff
  QmitkIGITimerBasedThread::run();

  ok = disconnect(this, SIGNAL(SignalBump()), this, SLOT(DoWakeUp()));
  assert(ok);
  ok = disconnect(this, SIGNAL(SignalCompress(unsigned int, unsigned int*)), this, SLOT(DoCompressFrame(unsigned int, unsigned int*)));
  assert(ok);
  ok = disconnect(this, SIGNAL(SignalStopCompression()), this, SLOT(DoStopCompression()));
  assert(ok);
  ok = disconnect(this, SIGNAL(SignalGetRGBAImage(unsigned int, IplImage**, unsigned int*)), this, SLOT(DoGetRGBAImage(unsigned int, IplImage**, unsigned int*)));
  assert(ok);
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::DoWakeUp()
{
  OnTimeoutImpl();
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::OnTimeoutImpl()
{
  QMutexLocker    l(&lock);

  if (current_state == QmitkIGINVidiaDataSourceImpl::FAILED)
  {
    return;
  }
  if (current_state == QmitkIGINVidiaDataSourceImpl::DEAD)
  {
    return;
  }

  // not much to do in case of playback. it's all synchronous on the gui thread.
  if (current_state == PLAYBACK)
  {
    return;
  }

  if (current_state == QmitkIGINVidiaDataSourceImpl::PRE_INIT)
  {
    oglwin->makeCurrent();
    current_state = QmitkIGINVidiaDataSourceImpl::HW_ENUM;

    // side note: we could try to avoid a race-condition of PRE_INIT-make-current and signal delivery
    // by connecting the signals only after HW_ENUM, when we know that it is safe to deliver them.
    // however, i dont know how well that mixes with signal drop-out recovery and repeated HW_ENUM,
    // which would then connect signals multiple times.
  }

  // make sure nobody messes around with contexts
  assert(QGLContext::currentContext() == oglwin->context());

  if (current_state == QmitkIGINVidiaDataSourceImpl::HW_ENUM)
  {
    try
    {
      // libvideo does its own glew init, so we can get cracking straight away
      InitVideo();

      // once we have an input setup
      //  grab at least one frame (not quite sure why)
      if (sdiin)
      {
        sdiin->capture();

        // make sure we are checking with twice the frame rate.
        // sampling theorem and stuff
        SetInterval((unsigned int) std::max(1, (int) (500.0f / format.get_refreshrate())));
      }
    }
    // getting an exception means something is broken
    // during normal operation this should never happen
    //  even if there's no hardware or signal
    catch (const std::exception& e)
    {
      state_message = std::string("Failed: ") + e.what();
      current_state = QmitkIGINVidiaDataSourceImpl::FAILED;
      // no point trying check often, it's not going to recover
      SetInterval(1000);
      return;
    }
    catch (...)
    {
      state_message = "Failed";
      current_state = QmitkIGINVidiaDataSourceImpl::FAILED;
      // no point trying check often, it's not going to recover
      SetInterval(1000);
      return;
    }
  }

  if (!HasHardware())
  {
    state_message = "No SDI hardware";
    // no hardware then nothing to do
    current_state = QmitkIGINVidiaDataSourceImpl::DEAD;
    // no point trying check often, it's not going to recover
    SetInterval(1000);
    return;
  }

  if (!HasInput())
  {
    state_message = "No input signal";
    // no signal, try again next round
    current_state = QmitkIGINVidiaDataSourceImpl::HW_ENUM;
    // dont re-setup too quickly
    SetInterval(500);
    return;
  }

  // if we get to here then we should be good to go!
  current_state = QmitkIGINVidiaDataSourceImpl::RUNNING;

  try
  {
    bool hasframe = sdiin->has_frame();
    // note: has_frame() will not throw an exception in case setup is broken

    // make sure we try to capture a frame if the previous one was too long ago.
    // that will check for errors and throw an exception if necessary, which will then allow us to restart.
    if ((timeGetTime() - m_LastSuccessfulFrame) > 1000)
      hasframe = true;

    if (hasframe)
    {
      // note: capture() will block for a frame to arrive
      // that's why we have hasframe above
      video::FrameInfo fi = sdiin->capture();
      // in case of partial transfers we'll get the zero sequence number.
      if (fi.sequence_number != 0)
      {
        m_LastSuccessfulFrame = timeGetTime();

        // keep the most recent set of texture ids around
        // this is mainly for the preview window
        for (int i = 0; i < 4; ++i)
        {
          textureids[i] = sdiin->get_texture_id(i, -1);
        }

        igtl::TimeStamp::Pointer timeCreated = igtl::TimeStamp::New();
        fi.id = GetTimeInNanoSeconds(timeCreated);//->GetTimeUint64();

        int newest_slot = sdiin->get_current_ringbuffer_slot();
        // whatever we had in this slot is now obsolete
        BOOST_TYPEOF(slot2sn_map)::iterator oldsni = slot2sn_map.find(newest_slot);
        if (oldsni != slot2sn_map.end())
        {
          BOOST_TYPEOF(sn2slot_map)::iterator oldsloti = sn2slot_map.find(oldsni->second);
          if (oldsloti != sn2slot_map.end())
          {
            sn2slot_map.erase(oldsloti);
          }
          slot2sn_map.erase(oldsni);
        }
        slot2sn_map[newest_slot] = fi;
        sn2slot_map[fi] = newest_slot;

        state_message = "Grabbing";
      }
      else
      {
        // in case of partial captures hasframe will be true but we wont get new sequence numbers.
        // this is effectively a brown-out. not sure how to handle it actually.
        // because we'd want the capture bits to recover with the original number of streams. unless of course
        // there was a genuine change of camera etc.
        if ((timeGetTime() - m_LastSuccessfulFrame) > 2000)
          throw "trigger handler below";
      }
    }
  }
  // capture() might throw if the capture setup has become invalid
  // e.g. a mode change or signal lost
  catch (...)
  {
    state_message = "Glitched out";
    current_state = QmitkIGINVidiaDataSourceImpl::HW_ENUM;
    // dont re-setup too quickly
    SetInterval(500);
    // whatever sequence numbers we had in the ringbuffer are now obsolete.
    // if we dont explicitly delete these then QmitkIGINVidiaDataSource::GrabData() will get mightily confused.
    sn2slot_map.clear();
    slot2sn_map.clear();
    return;
  }
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::Reset()
{
  QMutexLocker    l(&lock);

  // dont forget to remove any stale sequence numbers.
  sn2slot_map.clear();
  slot2sn_map.clear();

  state_message = "Initialising";

  current_state = PRE_INIT;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::DoGetRGBAImage(unsigned int sequencenumber, IplImage** img, unsigned int* streamcountinimg)
{
  QMutexLocker    l(&lock);

  // temp obj to search for the correct slot
  video::FrameInfo    fi = {0};
  fi.sequence_number = sequencenumber;

  BOOST_AUTO(sni, sn2slot_map.lower_bound(fi));
  // we need to check whether the request sequence number is still valid.
  // there may have been a capture reset.
  if (sni == sn2slot_map.end())
  {
    *img = 0;
    *streamcountinimg = 0;
    return;
  }


  // make sure nobody messes around with contexts.
  // notice that we put this check here and not at the top so that the above check for
  // sequence number bails out if there are no captured frames.
  // this is necessary because otherwise there could be a race condition between this-sdi-thread
  // and gui-update where gui-update tries a GetRGBAImage() before sdi-thread has had a chance to 
  // check for InitVideo() with the corresponding ogl-make-current.
  assert(QGLContext::currentContext() == oglwin->context());

  int w = sdiin->get_width();
  int h = sdiin->get_height();

  IplImage* frame = cvCreateImage(cvSize(w, h * streamcount), IPL_DEPTH_8U, 4);
  // mark layout as rgba instead of the opencv-default bgr
  std::memcpy(&frame->channelSeq[0], "RGBA", 4);

  ReadbackRGBA(frame->imageData, frame->widthStep, frame->width, frame->height, sni->second);

  *img = frame;
  *streamcountinimg = streamcount;
}


//-----------------------------------------------------------------------------
std::pair<IplImage*, int> QmitkIGINVidiaDataSourceImpl::GetRGBAImage(unsigned int sequencenumber)
{
  IplImage*     img = 0;
  unsigned int  streamcount = 0;
  emit SignalGetRGBAImage(sequencenumber, &img, &streamcount);

  return std::make_pair(img, streamcount);
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::DoCompressFrame(unsigned int sequencenumber, unsigned int* frameindex)
{
  QMutexLocker    l(&lock);

  // make sure nobody messes around with contexts
  assert(QGLContext::currentContext() == oglwin->context());


  if (compressor == 0)
  {
    CUresult r = cuCtxPushCurrent(cuContext);
    // die straight away
    if (r != CUDA_SUCCESS)
    {
      *frameindex = 0;
      return;
    }

    // also keep sdi logs
    sdiin->flush_log();
    sdiin->set_log_filename(m_CompressionOutputFilename + ".sdicapture.log");

    // when we get a new compressor we want to start counting from zero again
    m_NumFramesCompressed = 0;

    // it's unlikely that we'll ever catch an exception here.
    // all the other bits around the sdi data source would start failing earlier
    // if there's something wrong and we'd never even get here.
    try
    {
      compressor = new video::Compressor(sdiin->get_width(), sdiin->get_height(), format.refreshrate * streamcount, m_CompressionOutputFilename);
    }
    catch (...)
    {
      // die straight away
      CUcontext ctx;
      cuCtxPopCurrent(&ctx);
      *frameindex = 0;
      return;
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
    {
      *frameindex = 0;
      return;
    }
    assert(ctx == cuContext);
  }

  // find out which ringbuffer slot the request sequence number is in, if any
  video::FrameInfo  fi = {0};
  fi.sequence_number = sequencenumber;
  BOOST_TYPEOF(sn2slot_map)::const_iterator sloti = sn2slot_map.find(fi);
  if (sloti != sn2slot_map.end())
  {
    // sanity check
    assert(slot2sn_map.find(sloti->second)->second.sequence_number == sequencenumber);

    // compress each stream
    for (int i = 0; i < streamcount; ++i)
    {
      int tid = sdiin->get_texture_id(i, sloti->second);
      assert(tid != 0);
      // would need to do prepare() only once
      // but more often is ok too
      compressor->preparetexture(tid);
      compressor->compresstexture(tid);

      m_NumFramesCompressed++;
    }

    *frameindex = m_NumFramesCompressed;
  }
  else
  {
    // i want to know how often this happens, really...
    std::cerr << "Debug: sdi compressor: requested sn that is no longer available" << std::endl;
    assert(false);
    *frameindex = 0;
  }

}


//-----------------------------------------------------------------------------
unsigned int QmitkIGINVidiaDataSourceImpl::CompressFrame(unsigned int sequencenumber)
{
  unsigned int frameindex = 0;
  emit SignalCompress(sequencenumber, &frameindex);
  return frameindex;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::DoStopCompression()
{
  QMutexLocker    l(&lock);

  // make sure nobody messes around with contexts
  assert(QGLContext::currentContext() == oglwin->context());

  sdiin->flush_log();

  delete compressor;
  compressor = 0;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::StopCompression()
{
  emit SignalStopCompression();
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceImpl::TryPlayback(const std::string& filename)
{
  // FIXME: we still need to queue this onto the sdi thread!
  //        because we need a cuda context (even though we are using purevideo)
  QMutexLocker    l(&lock);

  delete decompressor;
  decompressor = 0;

  try
  {
    decompressor = new video::Decompressor(filename);

    std::ifstream   indexfile((filename + ".nalindex").c_str());
    if (indexfile.good())
    {

      indexfile.close();
    }
    else
    {
      // dont leave an open file handle lingering around
      indexfile.close();

      bool ok = decompressor->recover_index();
      if (!ok)
      {
        throw std::runtime_error("No index found and cannot recover one from stream.");
      }
      // try to write it to file so we dont have to do this repeatedly
      std::ofstream   recoveredindexfile((filename + ".nalindex").c_str());
      if (recoveredindexfile.is_open())
      {
        for (unsigned int f = 0; ; ++f)
        {
          unsigned __int64      offset = 0;
          video::FrameType::FT  type;

          ok = decompressor->get_index(f, &offset, &type);
          if (!ok)
          {
            break;
          }
          recoveredindexfile << f << ' ' << offset << ' ' << type << std::endl;
        }
        recoveredindexfile.close();
      }
    }
    
  }
  catch (const std::exception& e)
  {
    std::cerr << "Decompression init failed: " << e.what() << std::endl;
    throw;
  }
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSourceImpl::SequenceNumberComparator::operator()(const video::FrameInfo& a, const video::FrameInfo& b) const
{
  return a.sequence_number < b.sequence_number;
}
