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

  video::FrameInfo      last_successful_frame;

  // we keep our own copy of the texture ids (instead of relying on sdiin)
  //  so that another thread can easily get these
  // SDIInput is actively enforcing an opengl context check that's incompatible
  //  with the current threading situation
  int                     textureids[4];

  volatile IplImage*      copyoutasap;
  QWaitCondition          copyoutfinished;
  QMutex                  copyoutmutex;
  int                     copyoutslot;      // which slot in the ringbuffer

  // maps sequence numbers to ringbuffer slots
  std::map<unsigned int, int>   sn2slot_map;
  // maps ringbuffer slots to sequence numbers
  std::map<int, unsigned int>   slot2sn_map;

public:
  QmitkIGINVidiaDataSourceImpl()
    : sdidev(0), sdiin(0), streamcount(0), oglwin(0), oglshare(0), lock(QMutex::Recursive), 
      current_state(PRE_INIT), copyoutasap(0)
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
          break;

        format = f;
      }

      if (format.format != video::StreamFormat::PF_NONE)
      {
        sdiin = new video::SDIInput(sdidev, video::SDIInput::DO_NOTHING_SPECIAL, format.get_refreshrate());
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
, m_Pimpl(new QmitkIGINVidiaDataSourceImpl)
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

  this->InitializeAndRunGrabbingThread(20);
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource::~QmitkIGINVidiaDataSource()
{
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


  // FIXME: currently the grabbing thread's sole purpose is to call into this method
  //        so we could just block it here with a while loop and do our capture stuff

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
      m_Pimpl->check_video();

      // once we have an input setup
      //  grab at least one frame, there seems to be some glitch in the driver
      //  where has_frame() always returns false
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
    
    if (hasframe)
    {
      // note: capture() will block for a frame to arrive
      // that's why we have hasframe above
      video::FrameInfo fi = m_Pimpl->sdiin->capture();

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

      // Aim of this method is to do something like when a NiftyLink message comes in.
      mitk::IGINVidiaDataType::Pointer wrapper = mitk::IGINVidiaDataType::New();

      wrapper->SetValues(1, fi.sequence_number, fi.arrival_time);

      wrapper->SetDataSource("QmitkIGINVidiaDataSource");
      wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(timeCreated));
      wrapper->SetDuration(1000000000); // nanoseconds

      this->AddData(wrapper.GetPointer());

      this->SetStatus("Grabbing");

      // We signal every time we receive data, rather than at the GUI refresh rate, otherwise video looks very odd.
      //emit UpdateDisplay();
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
}


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

        int   subimagheight = frame.first->height / streamcount;
        IplImage  subimg;
        cvInitImageHeader(&subimg, cvSize((int) frame.first->width, subimagheight), IPL_DEPTH_8U, frame.first->nChannels);
        cvSetData(&subimg, &frame.first->imageData[i * subimagheight * frame.first->widthStep], frame.first->widthStep);

        // Check if we already have an image on the node.
        // We dont want to create a new one in that case (there is a lot of stuff going on
        // for allocating a new image).
        mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());
        if (imageInNode.IsNull())
        {
          mitk::Image::Pointer convertedImage = this->CreateMitkImage(&subimg);
          node->SetData(convertedImage);
        }
        else
        {
          // FIXME: check size of image that is already attached to data node!

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
  bool success = false;
  outputFileName = "";

  mitk::IGINVidiaDataType::Pointer dataType = static_cast<mitk::IGINVidiaDataType*>(data);
  if (dataType.IsNotNull())
  {
    // To do.
  }

  return success;
}

int QmitkIGINVidiaDataSource::GetNumberOfStreams()
{
  return 0;
}

int QmitkIGINVidiaDataSource::GetCaptureWidth()
{
  if (m_Pimpl == 0)
    return 0;

  video::StreamFormat format = m_Pimpl->get_format();
  return format.get_width();
}

int QmitkIGINVidiaDataSource::GetCaptureHeight()
{
  if (m_Pimpl == 0)
    return 0;

  video::StreamFormat format = m_Pimpl->get_format();
  return format.get_height();
}

int QmitkIGINVidiaDataSource::GetRefreshRate()
{
  if (m_Pimpl == 0)
    return 0;

  video::StreamFormat format = m_Pimpl->get_format();
  return format.get_refreshrate();
}

QGLWidget* QmitkIGINVidiaDataSource::GetCaptureContext()
{
    assert(m_Pimpl != 0);
    assert(m_Pimpl->oglshare != 0);
    return m_Pimpl->oglshare;
}

int QmitkIGINVidiaDataSource::GetTextureId(int stream)
{
    return m_Pimpl->get_texture_id(stream);
}
