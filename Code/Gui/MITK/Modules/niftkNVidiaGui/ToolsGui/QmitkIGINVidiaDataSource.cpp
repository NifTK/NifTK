/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGINVidiaDataSource.h"
#include "mitkIGINVidiaDataType.h"
#include <igtlTimeStamp.h>
#include <QTimer>
#include <QCoreApplication>
#include <QGLContext>
#include <QMutex>
#include <QGLWidget>
#include <QWaitCondition>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include "video/sdiinput.h"


struct QmitkIGINVidiaDataSourceImpl// : public QThread
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

  volatile bool           stop_asap;

  // any access to the capture bits needs to be locked
  mutable QMutex          lock;
  video::SDIDevice*       sdidev;
  video::SDIInput*        sdiin;
  video::StreamFormat     format;
  int                     streamcount;

  video::FrameInfo      last_successful_frame;

  // we keep our own copy of the texture ids (instead of relying on sdiin)
  //  so that another thread can easily get these
  // SDIInput is actively enforcing an opengl context check that's incompatible
  //  with the current threading situation
  int                     textureids[4];

  volatile IplImage*      copyoutasap;
  QWaitCondition          copyoutfinished;
  QMutex                  copyoutmutex;

public:
  QmitkIGINVidiaDataSourceImpl()
    : sdidev(0), sdiin(0), streamcount(0), oglwin(0), oglshare(0), lock(QMutex::Recursive), 
      current_state(PRE_INIT), copyoutasap(0), stop_asap(false)
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

    last_successful_frame.sequence_number = 0;
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

      delete sdiin;
      // we do not own sdidev!
      sdidev = 0;
    
      if (ctx)
        ctx->makeCurrent();
    }
    catch (...)
    {
        std::cerr << "sdi cleanup threw exception" << std::endl;
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
  void check_video()
  {
    // make sure nobody messes around with contexts
    assert(QGLContext::currentContext() == oglwin->context());

    // we do not own the device!
    sdidev = 0;
    // but we gotta clear up this one
    delete sdiin;
    sdiin = 0;

    // find our capture card
    for (int i = 0; ; ++i)
    {
      video::SDIDevice*	d = video::SDIDevice::get_device(i);
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
          break;

        format = f;
      }

      if (format.format != video::StreamFormat::PF_NONE)
      {
        sdiin = new video::SDIInput(sdidev, video::SDIInput::STACK_FIELDS);

        // FIXME: these might not be constant throughout a single capture sessions!
        for (int i = 0; i < 4; ++i)
          textureids[i] = sdiin->get_texture_id(i);
      }
    }

    // assuming everything went fine
    //  we now have texture objects that will receive video data everytime we call capture()
  }
    
/*
        // scoping for lock
        {
          // other threads might supply us with a new pointer!
          QMutexLocker    l(&lock);
          
          if (copyoutasap)
          {
            

            // and reset pointer
            // so that we dont do it again unless explicitly requested
            copyoutasap = 0;

            copyoutfinished.wakeOne();
          }
        }
*/


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
      int texid = sdiin->get_texture_id(i);
      if (texid == 0)
        break;

      // while we have the lock, texture dimensions are not going to change
     // if ((i * dim.second) >= height)
     //   return;
      // remaining buffer too small?
      if (height - (i * dim.second) < dim.second)
        return;

      char*   subbuf = &buffer[i * dim.second * bufferpitch];

      glBindTexture(GL_TEXTURE_2D, texid);
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, subbuf);
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

//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource::QmitkIGINVidiaDataSource()
: pimpl(new QmitkIGINVidiaDataSourceImpl)
{
  this->SetName("QmitkIGINVidiaDataSource");
  this->SetType("Frame Grabber");
  this->SetDescription("NVidia SDI");
  this->SetStatus("Initialising...");

  this->InitializeAndRunGrabbingThread(20);
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource::~QmitkIGINVidiaDataSource()
{
  this->StopCapturing();

  pimpl->stop_asap = true;
  //pimpl->wait(5000);
  delete pimpl;
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
  // To do.
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::StopCapturing()
{
  // To do.
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::IsCapturing()
{
  bool result = false;

  // To do.

  return result;
}


std::pair<IplImage*, int> QmitkIGINVidiaDataSource::get_rgb_image()
{
  // i dont like this way of unstructured locking
  pimpl->lock.lock();
  // but the waitcondition stuff doesnt work otherwise

  std::pair<int, int>   imgdim = pimpl->get_capture_dimensions();
  int                   streamcount = pimpl->get_stream_count();

  IplImage* frame = cvCreateImage(cvSize(imgdim.first, imgdim.second * streamcount), IPL_DEPTH_8U, 3);
  // mark layout as rgb instead of the opencv-default bgr
  std::memcpy(&frame->channelSeq[0], "RGB\0", 4);

  pimpl->copyoutasap = frame;

  pimpl->copyoutmutex.lock();
  // until here, capture thread would be stuck waiting for the lock
  pimpl->lock.unlock();

  pimpl->copyoutfinished.wait(&pimpl->copyoutmutex);
  pimpl->copyoutmutex.unlock();
  return std::make_pair(frame, streamcount);

  // failed somewhere
//  cvReleaseImage(&frame);
//  return std::make_pair((IplImage*) 0, 0);
}

//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::GrabData()
{
  assert(pimpl != 0);

  assert(m_GrabbingThread == (QmitkIGILocalDataSourceGrabbingThread*) QThread::currentThread());


  // FIXME: currently the grabbing thread's sole purpose is to call into this method
  //        so we could just block it here with a while loop and do our capture stuff

  QMutexLocker    l(&pimpl->lock);

  if (pimpl->current_state == QmitkIGINVidiaDataSourceImpl::FAILED)
  {
    return;
  }
  if (pimpl->current_state == QmitkIGINVidiaDataSourceImpl::DEAD)
  {
    return;
  }

  if (pimpl->current_state == QmitkIGINVidiaDataSourceImpl::PRE_INIT)
  {
    pimpl->oglwin->makeCurrent();
    pimpl->current_state = QmitkIGINVidiaDataSourceImpl::HW_ENUM;
  }

  // make sure nobody messes around with contexts
  assert(QGLContext::currentContext() == pimpl->oglwin->context());

  if (pimpl->current_state == QmitkIGINVidiaDataSourceImpl::HW_ENUM)
  {
    try
    {
      // libvideo does its own glew init, so we can get cracking straight away
      pimpl->check_video();

      // once we have an input setup
      //  grab at least one frame, there seems to be some glitch in the driver
      //  where has_frame() always returns false
      if (pimpl->sdiin)
      {
        pimpl->sdiin->capture();
      }
    }
    // getting an exception means something is broken
    // during normal operation this should never happen
    //  even if there's no hardware or signal
    catch (const std::exception& e)
    {
      this->SetStatus(std::string("Failed: ") + e.what());
      pimpl->current_state = QmitkIGINVidiaDataSourceImpl::FAILED;
      return;
    }
    catch (...)
    {
      this->SetStatus("Failed");
      pimpl->current_state = QmitkIGINVidiaDataSourceImpl::FAILED;
      return;
    }
  }

  if (!pimpl->has_hardware())
  {
    this->SetStatus("No SDI hardware");
    // no hardware then nothing to do
    pimpl->current_state = QmitkIGINVidiaDataSourceImpl::DEAD;
    return;
  }

  if (!pimpl->has_input())
  {
    this->SetStatus("No input signal");
    // no signal, try again next round
    pimpl->current_state = QmitkIGINVidiaDataSourceImpl::HW_ENUM;
    return;
  }

  // if we get to here then we should be good to go!
  pimpl->current_state = QmitkIGINVidiaDataSourceImpl::RUNNING;

  try
  {
    bool hasframe = pimpl->sdiin->has_frame();
    // note: has_frame() will not throw an exception in case setup is broken
    
    if (hasframe)
    {
      // note: capture() will block for a frame to arrive
      // that's why we have hasframe above
      video::FrameInfo fi = pimpl->sdiin->capture();

      igtl::TimeStamp::Pointer timeCreated = igtl::TimeStamp::New();

      // Aim of this method is to do something like when a NiftyLink message comes in.
      mitk::IGINVidiaDataType::Pointer wrapper = mitk::IGINVidiaDataType::New();

      wrapper->set_values(1, fi.sequence_number, fi.arrival_time);

      wrapper->SetDataSource("QmitkIGINVidiaDataSource");
      wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(timeCreated));
      wrapper->SetDuration(1000000000); // nanoseconds

      this->AddData(wrapper.GetPointer());

      this->SetStatus("Grabbing");

      if (pimpl->copyoutasap)
      {
        pimpl->readback_rgb(pimpl->copyoutasap->imageData, pimpl->copyoutasap->widthStep, pimpl->copyoutasap->width, pimpl->copyoutasap->height);
        pimpl->copyoutasap = 0;
        pimpl->copyoutfinished.wakeOne();
      }
    }
  }
  // capture() might throw if the capture setup has become invalid
  // e.g. a mode change or signal lost
  catch (...)
  {
    this->SetStatus("Glitched out");
    pimpl->current_state = QmitkIGINVidiaDataSourceImpl::HW_ENUM;
    return;
  }

  // We signal every time we receive data, rather than at the GUI refresh rate, otherwise video looks very odd.
  emit UpdateDisplay();
}


bool QmitkIGINVidiaDataSource::Update(mitk::IGIDataType* data)
{
  bool result = false;

  mitk::IGINVidiaDataType::Pointer dataType = static_cast<mitk::IGINVidiaDataType*>(data);
  if (dataType.IsNotNull())
  {
    // FIXME: need to pass in the requested timestamp etc!

    // one massive image, with all streams stacked in
    std::pair<IplImage*, int> frame = get_rgb_image();
    // if copy-out failed then capture setup is broken, e.g. someone unplugged a cable
    if (frame.first)
    {
      // max 4 streams
      const int streamcount = frame.second;
      for (int i = 0; i < streamcount; ++i)
      {
        std::ostringstream  nodename;
        nodename << "NVIDIA SDI stream " << i;
        std::vector<mitk::DataNode::Pointer> dataNode = this->GetDataNode(nodename.str());
        if (dataNode.size() != 1)
        {
          MITK_ERROR << "QmitkIGINVidiaDataSource only supports a single video image per feed" << std::endl;
          this->SetStatus("Failed");
          cvReleaseImage(&frame.first);
          return false;
        }
        mitk::DataNode::Pointer node = dataNode[0];

        // FIXME: chop!
        mitk::Image::Pointer convertedImage = this->CreateMitkImage(frame.first);
        mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());
        if (imageInNode.IsNull())
        {
          node->SetData(convertedImage);
        }
        else
        {
          try
          {
            mitk::ImageReadAccessor readAccess(convertedImage, convertedImage->GetVolumeData(0));
            const void* cPointer = readAccess.GetData();

            mitk::ImageWriteAccessor writeAccess(imageInNode);
            void* vPointer = writeAccess.GetData();

            assert(frame.first->nChannels == 3);
            std::memcpy(vPointer, cPointer, frame.first->width * frame.first->height * 3);
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



  return result;
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::SaveData(mitk::IGIDataType* data, std::string& outputFileName)
{
  bool success = false;
  outputFileName = "";

  mitk::IGINVidiaDataType::Pointer dataType = static_cast<mitk::IGINVidiaDataType*>(data);
  if (dataType.IsNotNull())
  {
    // To do.
  }

  return success;
}

int QmitkIGINVidiaDataSource::get_number_of_streams()
{
    return 0;
}

int QmitkIGINVidiaDataSource::get_capture_width()
{
  if (pimpl == 0)
    return 0;

  video::StreamFormat format = pimpl->get_format();
  return format.get_width();
}

int QmitkIGINVidiaDataSource::get_capture_height()
{
  if (pimpl == 0)
    return 0;

  video::StreamFormat format = pimpl->get_format();
  return format.get_height();
}

int QmitkIGINVidiaDataSource::get_refresh_rate()
{
  if (pimpl == 0)
    return 0;

  video::StreamFormat format = pimpl->get_format();
  return format.get_refreshrate();
}

QGLWidget* QmitkIGINVidiaDataSource::get_capturecontext()
{
    assert(pimpl != 0);
    assert(pimpl->oglshare != 0);
    return pimpl->oglshare;
}

int QmitkIGINVidiaDataSource::get_texture_id(int stream)
{
    return pimpl->get_texture_id(stream);
}
