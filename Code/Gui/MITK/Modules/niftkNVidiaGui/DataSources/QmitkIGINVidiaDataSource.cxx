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
#include <QGLWidget>
#include <QWaitCondition>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include "QmitkIGINVidiaDataSourceImpl.h"


// note the trailing space
const char* QmitkIGINVidiaDataSource::s_NODE_NAME = "NVIDIA SDI stream ";


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource::QmitkIGINVidiaDataSource(mitk::DataStorage* storage)
: QmitkIGILocalDataSource(storage)
, m_Pimpl(new QmitkIGINVidiaDataSourceImpl), m_MipmapLevel(0), m_MostRecentSequenceNumber(1)
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

  // needs to match GUI combobox's default!
  m_Pimpl->SetFieldMode((video::SDIInput::InterlacedBehaviour) STACK_FIELDS);
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

  m_Pimpl->SetFieldMode((video::SDIInput::InterlacedBehaviour) b);
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
  m_MostRecentSequenceNumber = 1;
  //m_Pimpl->Reset();
  m_Pimpl->start();
  this->InitializeAndRunGrabbingThread(20);
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::StopCapturing()
{
  m_Pimpl->ForciblyStop();
  StopGrabbingThread();
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::IsCapturing()
{
  bool result = m_Pimpl->isRunning();
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::GrabData()
{
  assert(m_Pimpl != 0);
  
  //should not be necessary anymore
  //assert(m_GrabbingThread == (QmitkIGILocalDataSourceGrabbingThread*) QThread::currentThread());

  // always update status message
  this->SetStatus(m_Pimpl->GetStateMessage());

  if (m_Pimpl->IsRunning())
  {
    video::FrameInfo  sn = {0};
    sn.sequence_number = m_MostRecentSequenceNumber;

    while (sn.sequence_number != 0)
    {
      sn = m_Pimpl->GetNextSequenceNumber(m_MostRecentSequenceNumber);
      if (sn.sequence_number != 0)
      {
        mitk::IGINVidiaDataType::Pointer wrapper = mitk::IGINVidiaDataType::New();

        wrapper->SetValues(m_Pimpl->GetCookie(), sn.sequence_number, sn.arrival_time);
        wrapper->SetTimeStampInNanoSeconds(sn.id);
        wrapper->SetDuration(this->m_TimeStampTolerance); // nanoseconds
        m_MostRecentSequenceNumber = sn.sequence_number;

        // under certain circumstances, this might call back into SaveData()
        this->AddData(wrapper.GetPointer());
      }
    }
  }

#ifdef JOHANNES_HAS_FIXED_RECORDING
  // because there's currently no notification when the user clicked stop-record
  // we need to check this way to clean up the compressor.
  if (m_Pimpl->m_WasSavingMessagesPreviously && !GetSavingMessages())
  {
    delete m_Pimpl->compressor;
    m_Pimpl->compressor = 0;
    m_Pimpl->m_WasSavingMessagesPreviously = false;

    m_FrameMapLogFile.close();
  }
#endif
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::Update(mitk::IGIDataType* data)
{
  bool result = true;

  mitk::IGINVidiaDataType::Pointer dataType = static_cast<mitk::IGINVidiaDataType*>(data);
  if (dataType.IsNotNull())
  {
    if (m_Pimpl->GetCookie() == dataType->GetCookie())
    {
      // one massive image, with all streams stacked in
      std::pair<IplImage*, int> frame = m_Pimpl->GetRgbaImage(dataType->GetSequenceNumber());
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
  //assert(m_GrabbingThread == (QmitkIGILocalDataSourceGrabbingThread*) QThread::currentThread());

  bool success = false;
  outputFileName = "";

  mitk::IGINVidiaDataType::Pointer dataType = static_cast<mitk::IGINVidiaDataType*>(data);
  if (dataType.IsNotNull())
  {
#ifdef JOHANNES_HAS_FIXED_RECORDING
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

      std::string directoryPath = this->m_SavePrefix + '/' + "QmitkIGINVidiaDataSource";
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
#endif
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

  return m_Pimpl->GetStreamCount();
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSource::GetCaptureWidth()
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  video::StreamFormat format = m_Pimpl->GetFormat();
  return format.get_width();
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSource::GetCaptureHeight()
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  video::StreamFormat format = m_Pimpl->GetFormat();
  return format.get_height();
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSource::GetRefreshRate()
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  video::StreamFormat format = m_Pimpl->GetFormat();
  return format.get_refreshrate();
}


//-----------------------------------------------------------------------------
QGLWidget* QmitkIGINVidiaDataSource::GetCaptureContext()
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  return m_Pimpl->GetCaptureContext();
}


//-----------------------------------------------------------------------------
int QmitkIGINVidiaDataSource::GetTextureId(int stream)
{
  if (m_Pimpl == 0)
  {
    return 0;
  }

  return m_Pimpl->GetTextureId(stream);
}
