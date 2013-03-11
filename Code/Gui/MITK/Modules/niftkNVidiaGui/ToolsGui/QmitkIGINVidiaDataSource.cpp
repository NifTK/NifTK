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
#include <mitkOpenCVToMitkImageFilter.h>

#include "video/sdiinput.h"


struct QmitkIGINVidiaDataSourceImpl
{
  video::SDIDevice*       sdidev;
  video::SDIInput*        sdiin;
  video::StreamFormat     format;
  int                     streamcount;

  // all the sdi stuff needs an opengl context
  //  so we'll create our own
  QGLContext*             oglctx;
  QGLWidget*              oglwin;
  mutable QMutex          lock;

public:
  QmitkIGINVidiaDataSourceImpl()
    : sdidev(0), sdiin(0), streamcount(0), oglctx(0), oglwin(0), lock(QMutex::Recursive)
  {
  }


  void init()
  {
    QMutexLocker    l(&lock);

    // we dont need much flags, there's no actual rendering on this context
    //  (for now)
    oglwin = new QGLWidget(0, 0, Qt::WindowFlags(Qt::Window | Qt::FramelessWindowHint));
    oglwin->hide();
    assert(oglwin->isValid());
    oglwin->makeCurrent();

    // libvideo does its own glew init, so we can get cracking straight away

    try
    {
      check_video();
    }
    catch (...)
    {
      // FIXME: need to report this back to gui somehow
      std::cerr << "Whoops" << std::endl;
    }
  }


protected:
  // FIXME: needs cfg param to decide which channel to capture, format, etc
  void check_video()
  {
    // make sure nobody messes around with contexts
//    assert(QGLContext::currentContext() == oglctx);

    QMutexLocker    l(&lock);

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


      }
    }

    // assuming everything went fine
    //  we now have texture objects that will receive video data everytime we call capture()
  }

public: // FIXME: should be protected
  bool capture()
  {
    QMutexLocker    l(&lock);
    if (sdiin)
    {
      try
      {
        oglwin->makeCurrent();
        sdiin->capture();

        return true;
      }
      catch (...)
      {
          return false;
      }
    }
    return false;
  }

public:
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

  // FIXME: one channel only
  bool copy_out_bgr(char* buffer, std::size_t bufferpitch, int width, int height)
  {
    QMutexLocker    l(&lock);

    if (sdiin)
    {
      // FIXME: this will change quite a bit once it runs in its own thread

      oglwin->makeCurrent();

      // unfortunately we have 3 bytes per pixel
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      assert((bufferpitch % 3) == 0);
      glPixelStorei(GL_PACK_ROW_LENGTH, bufferpitch / 3);

      glBindTexture(GL_TEXTURE_2D, sdiin->get_texture_id(0));
      glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, buffer);
      assert(glGetError() == GL_NO_ERROR);

      return true;
    }
    
    return false;
  }

  int get_texture_id(int stream)
  {
    QMutexLocker    l(&lock);
    if (sdiin == 0)
        return 0;
    return sdiin->get_texture_id(stream);
  }
};

//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource::QmitkIGINVidiaDataSource()
: m_Timer(NULL), pimpl(new QmitkIGINVidiaDataSourceImpl)
{
  this->SetName("QmitkIGINVidiaDataSource");
  this->SetType("Frame Grabber");
  this->SetDescription("NVidia SDI");
  this->SetStatus("Initialising...");

  // FIXME: this should be running in its own thread!
  pimpl->init();


  filter = mitk::OpenCVToMitkImageFilter::New();

  // FIXME: depends on number of active streams, etc
  m_ImageNode = mitk::DataNode::New();
  m_ImageNode->SetName("nvidia sdi input node");
  m_ImageNode->SetVisibility(true);
    

  this->StartCapturing();

  m_Timer = new QTimer();
  m_Timer->setInterval(20); // milliseconds
  m_Timer->setSingleShot(false);

  connect(m_Timer, SIGNAL(timeout()), this, SLOT(OnTimeout()));



  m_Timer->start();
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource::~QmitkIGINVidiaDataSource()
{
  this->StopCapturing();
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

IplImage* QmitkIGINVidiaDataSource::get_bgr_image()
{
  std::pair<int, int>   imgdim = pimpl->get_capture_dimensions();
  IplImage* frame = cvCreateImage(cvSize(imgdim.first, imgdim.second), IPL_DEPTH_8U, 3);

  bool ok = pimpl->copy_out_bgr(frame->imageData, frame->widthStep, frame->width, frame->height);
  if (ok)
    return frame;

  // failed somewhere
  cvReleaseImage(&frame);
  return 0;
}

//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::OnTimeout()
{
  assert(pimpl != 0);

  if (!pimpl->has_hardware())
  {
    this->SetStatus("No SDI hardware");
    return;
  }

  if (!pimpl->has_input())
  {
    this->SetStatus("No input signal");
    return;
  }

  bool captureok = pimpl->capture();

#if 0
  // FIXME: bad idea to use that mitk filter
  //        it only works with rgb, no alpha
  //        and it assumes layout is bgr, ie. desktop webcams dumping data to gdi
  IplImage* frame = get_bgr_image();
  // if copy-out failed then capture setup is broken, e.g. someone unplugged a cable
  if (frame)
  {
    filter->SetCopyBuffer(true);   // FIXME: dont know what that is supposed to do, if i set it to true we get massive mem leak
    filter->SetOpenCVImage(frame);
    filter->Update();
    m_Image = filter->GetOutput(0);
    if (!this->GetDataStorage()->Exists(m_ImageNode))
    {
      this->GetDataStorage()->Add(m_ImageNode);
    }
    m_ImageNode->SetData(m_Image);
    // disconnect temporary image?
    filter->SetOpenCVImage(0); 
    cvReleaseImage(&frame);
#else
  if (captureok)
  {
#endif
    this->SetStatus("Grabbing");

    igtl::TimeStamp::Pointer timeCreated = igtl::TimeStamp::New();
    timeCreated->GetTime();

    // Aim of this method is to do something like when a NiftyLink message comes in.
    mitk::IGINVidiaDataType::Pointer wrapper = mitk::IGINVidiaDataType::New();
    //wrapper->CloneImage(m_VideoSource->GetCurrentFrame()); either copy/clone the data or, just store some kind of frame count.
    wrapper->SetDataSource("QmitkIGINVidiaDataSource");
    wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(timeCreated));
    wrapper->SetDuration(1000000000); // nanoseconds

    this->AddData(wrapper.GetPointer());
  }
  else
    this->SetStatus("Failed");

  // We signal every time we receive data, rather than at the GUI refresh rate, otherwise video looks very odd.
  emit UpdateDisplay();
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
    assert(pimpl->oglwin != 0);
    return pimpl->oglwin;
}

int QmitkIGINVidiaDataSource::get_texture_id(int stream)
{
    return pimpl->get_texture_id(stream);
}
