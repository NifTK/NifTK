/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtCameraVideoDataSourceService.h"
#include "niftkQtCameraVideoDataType.h"
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataSourceUtils.h>
#include <niftkQImageConversion.h>
#include <mitkExceptionMacro.h>
#include <mitkImage.h>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <QDir>
#include <QList>
#include <QMutexLocker>
#include <QCamera>
#include <QCameraInfo>
#include "cameraframegrabber.h"

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker QtCameraVideoDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
QtCameraVideoDataSourceService::QtCameraVideoDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("QtVideo-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_Camera(nullptr)
, m_FrameId(0)
, m_Buffer(nullptr)
, m_BackgroundDeleteThread(nullptr)
{
  this->SetStatus("Initialising");

  QString deviceName = this->GetName();
  m_ChannelNumber = (deviceName.remove(0, 8)).toInt(); // Should match string QtVideo- above

  if(!properties.contains("name"))
  {
    mitkThrow() << "Video device name not specified!";
  }
  QString videoDeviceName = (properties.value("name")).toString();

  QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
  foreach (const QCameraInfo &cameraInfo, cameras)
  {
    if (cameraInfo.deviceName() == videoDeviceName)
    {
      m_Camera = new QCamera(cameraInfo);
    }
  }
  if (m_Camera == nullptr)
  {
    mitkThrow() << "Failed to create video source:" << videoDeviceName.toStdString();
  }
  m_Camera->setCaptureMode(QCamera::CaptureVideo);
  m_CameraFrameGrabber = new CameraFrameGrabber();
  m_Camera->setViewfinder(m_CameraFrameGrabber);

  bool ok = QObject::connect(m_CameraFrameGrabber, SIGNAL(frameAvailable(QImage)),
                             this, SLOT(OnFrameAvailable(QImage)));
  assert(ok);

  m_Camera->start();

  int defaultFramesPerSecond = 25;
  m_Buffer = niftk::IGIDataSourceBuffer::New(defaultFramesPerSecond * 2);

  // Set the interval based on desired number of frames per second.
  // So, 25 fps = 40 milliseconds.
  // However: If system slows down (eg. saving images), then Qt will
  // drop clock ticks, so in effect, you will get less than this.
  int intervalInMilliseconds = 1000 / defaultFramesPerSecond;
  this->SetTimeStampTolerance(intervalInMilliseconds*1000000*1.5 /* fudge factor*/);
  this->SetShouldUpdate(true);
  this->SetProperties(properties);

  m_BackgroundDeleteThread = new niftk::IGIDataSourceBackgroundDeleteThread(NULL, this);
  m_BackgroundDeleteThread->SetInterval(2000); // try deleting images every 2 seconds.
  m_BackgroundDeleteThread->start();
  if (!m_BackgroundDeleteThread->isRunning())
  {
    mitkThrow() << "Failed to start background deleting thread";
  }

  this->SetDescription("Local video source, (via QtCamera), e.g. web-cam.");
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
QtCameraVideoDataSourceService::~QtCameraVideoDataSourceService()
{
  if (m_Camera != nullptr)
  {
    m_Camera->stop();
    delete m_Camera;
  }
  if (m_CameraFrameGrabber != nullptr)
  {
    delete m_CameraFrameGrabber;
  }

  s_Lock.RemoveSource(m_ChannelNumber);

  m_BackgroundDeleteThread->ForciblyStop();
  delete m_BackgroundDeleteThread;
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  if (properties.contains("lag"))
  {
    int milliseconds = (properties.value("lag")).toInt();
    m_Buffer->SetLagInMilliseconds(milliseconds);

    MITK_INFO << "QtCameraVideoDataSourceService(" << this->GetName().toStdString()
              << "): Set lag to " << milliseconds << " ms.";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties QtCameraVideoDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  props.insert("lag", m_Buffer->GetLagInMilliseconds());

  MITK_INFO << "QtCameraVideoDataSourceService(:" << this->GetName().toStdString()
            << "):Retrieved current value of lag as " << m_Buffer->GetLagInMilliseconds();

  return props;
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataSourceService::OnFrameAvailable(const QImage &image)
{
  niftk::QtCameraVideoDataType::Pointer wrapper = niftk::QtCameraVideoDataType::New();
  wrapper->CloneImage(image);
  wrapper->SetTimeStampInNanoSeconds(this->GetTimeStampInNanoseconds());
  wrapper->SetFrameId(m_FrameId++);
  wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
  wrapper->SetShouldBeSaved(this->GetIsRecording());

  // Save synchronously.
  // This has the side effect that if saving is too slow,
  // the QTimers just won't keep up, and start missing pulses.
  if (this->GetIsRecording())
  {
    this->SaveItem(wrapper.GetPointer());
    this->SetStatus("Saving");
  }
  else
  {
    this->SetStatus("Grabbing");
  }

  // Putting this after the save, as we don't want to
  // add to the buffer in this grabbing thread, then the
  // m_BackgroundDeleteThread deletes the object while
  // we are trying to save the data.
  m_Buffer->AddToBuffer(wrapper.GetPointer());
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataSourceService::CleanBuffer()
{
  // Buffer itself should be threadsafe.
  m_Buffer->CleanBuffer();
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataType::IGITimeType lastTimeStamp)
{
  QMutexLocker locker(&m_Lock);

  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);

  m_Buffer->DestroyBuffer();

  QDir directory(this->GetPlaybackDirectory());
  if (directory.exists())
  {
    std::set<niftk::IGIDataType::IGITimeType> timeStamps;
    niftk::ProbeTimeStampFiles(directory, QString(".jpg"), timeStamps);
    m_PlaybackIndex = timeStamps;
  }
  else
  {
    assert(false);
  }
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataSourceService::StopPlayback()
{
  QMutexLocker locker(&m_Lock);

  m_PlaybackIndex.clear();
  m_Buffer->DestroyBuffer();

  IGIDataSource::StopPlayback();
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataSourceService::PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp)
{
  assert(this->GetIsPlayingBack());
  assert(!m_PlaybackIndex.empty()); // Should have failed probing if no data.

  // this will find us the timestamp right after the requested one
  std::set<niftk::IGIDataType::IGITimeType>::const_iterator i = m_PlaybackIndex.upper_bound(requestedTimeStamp);
  if (i != m_PlaybackIndex.begin())
  {
    --i;
  }
  if (i != m_PlaybackIndex.end())
  {
    if (!m_Buffer->Contains(*i))
    {
      std::ostringstream  filename;
      filename << this->GetPlaybackDirectory().toStdString() << '/' << (*i) << ".jpg";
/*
      IplImage* img = cvLoadImage(filename.str().c_str());
      if (img)
      {
        niftk::QtCameraVideoDataType::Pointer wrapper = niftk::QtCameraVideoDataType::New();
        wrapper->CloneImage(img);
        wrapper->SetTimeStampInNanoSeconds(*i);
        wrapper->SetFrameId(m_FrameId++);
        wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
        wrapper->SetShouldBeSaved(false);

        // Buffer itself should be threadsafe, so I'm not locking anything here.
        m_Buffer->AddToBuffer(wrapper.GetPointer());

        cvReleaseImage(&img);
      }
*/
    }
    this->SetStatus("Playing back");
  }
}


//-----------------------------------------------------------------------------
bool QtCameraVideoDataSourceService::ProbeRecordedData(niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                                                     niftk::IGIDataType::IGITimeType* lastTimeStampInStore)
{
  // zero is a suitable default value. it's unlikely that anyone recorded a legitime data set in the middle ages.
  niftk::IGIDataType::IGITimeType  firstTimeStampFound = 0;
  niftk::IGIDataType::IGITimeType  lastTimeStampFound  = 0;

  QDir directory(this->GetPlaybackDirectory());
  if (directory.exists())
  {
    std::set<niftk::IGIDataType::IGITimeType> timeStamps;
    niftk::ProbeTimeStampFiles(directory, QString(".jpg"), timeStamps);
    if (!timeStamps.empty())
    {
      firstTimeStampFound = *timeStamps.begin();
      lastTimeStampFound  = *(--(timeStamps.end()));
    }
  }

  if (firstTimeStampInStore)
  {
    *firstTimeStampInStore = firstTimeStampFound;
  }
  if (lastTimeStampInStore)
  {
    *lastTimeStampInStore = lastTimeStampFound;
  }

  return firstTimeStampFound != 0;
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataSourceService::SaveItem(niftk::IGIDataType::Pointer data)
{
  niftk::QtCameraVideoDataType::Pointer dataType = static_cast<niftk::QtCameraVideoDataType*>(data.GetPointer());
  if (dataType.IsNull())
  {
    mitkThrow() << "Failed to save QtCameraVideoDataType as the data received was the wrong type!";
  }
/*
  const IplImage* imageFrame = dataType->GetImage();
  if (imageFrame == NULL)
  {
    mitkThrow() << "Failed to save QtCameraVideoDataType as the image frame was NULL!";
  }

  QString directoryPath = this->GetRecordingDirectory();
  QDir directory(directoryPath);
  if (directory.mkpath(directoryPath))
  {
    QString fileName =  directoryPath + QDir::separator() + tr("%1.jpg").arg(data->GetTimeStampInNanoSeconds());
    bool success = cvSaveImage(fileName.toStdString().c_str(), imageFrame);
    if (!success)
    {
      mitkThrow() << "Failed to save QtCameraVideoDataType in cvSaveImage!";
    }
    data->SetIsSaved(true);
  }
  else
  {
    mitkThrow() << "Failed to save QtCameraVideoDataType as could not create " << directoryPath.toStdString();
  }
*/
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> QtCameraVideoDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;

  // Create default return status.
  // So, we always return at least 1 row.
  IGIDataItemInfo info;
  info.m_Name = this->GetName();
  info.m_FramesPerSecond = m_Buffer->GetFrameRate();
  info.m_IsLate = true;
  info.m_LagInMilliseconds = 0;
  infos.push_back(info);

  // This loads playback-data into the buffers, so must
  // come before the check for empty buffer.
  if (this->GetIsPlayingBack())
  {
    this->PlaybackData(time);
  }

  if (m_Buffer->GetBufferSize() == 0)
  {
    return infos;
  }

  if(m_Buffer->GetFirstTimeStamp() > time)
  {
    MITK_DEBUG << "QtCameraVideoDataSourceService::Update(), requested time is before buffer time! "
               << " Buffer size=" << m_Buffer->GetBufferSize()
               << ", time=" << time
               << ", firstTime=" << m_Buffer->GetFirstTimeStamp();
    return infos;
  }

  niftk::QtCameraVideoDataType::Pointer dataType =
    static_cast<niftk::QtCameraVideoDataType*>(m_Buffer->GetItem(time).GetPointer());

  if (dataType.IsNull())
  {
    MITK_DEBUG << "Failed to find data for time " << time
               << ", size=" << m_Buffer->GetBufferSize()
               << ", last=" << m_Buffer->GetLastTimeStamp() << std::endl;
    return infos;
  }

  // If we are not actually updating data, bail out.
  if (!this->GetShouldUpdate())
  {
    return infos;
  }

  mitk::DataNode::Pointer node = this->GetDataNode(this->GetName());
  if (node.IsNull())
  {
    mitkThrow() << "Can't find mitk::DataNode with name "
                << this->GetName().toStdString() << std::endl;
  }

  // Get Image from the dataType;
  const QImage* img = dataType->GetImage();
  if (img == NULL)
  {
    this->SetStatus("Failed Update");
    mitkThrow() << "Failed to extract QImage image from buffer!";
  }
  else
  {
    int imageDepth = img->depth();
    mitk::Image::Pointer convertedImage;
    if (img->format() == QImage::Format_RGB888
        || img->format() == QImage::Format_RGBA8888
        || img->format() == QImage::Format_Indexed8
        )
    {
      convertedImage = niftk::CreateMitkImage(img);
    }
    else
    {
      QImage tmp = img->convertToFormat(QImage::Format_RGB888);
      convertedImage = niftk::CreateMitkImage(&tmp);
      imageDepth = tmp.depth();
    }

    mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());
    if (imageInNode.IsNull())
    {
      // We remove and add to trigger the NodeAdded event,
      // which is not emmitted if the node was added with no data.
      this->GetDataStorage()->Remove(node);
      node->SetData(convertedImage);
      this->GetDataStorage()->Add(node);

      imageInNode = convertedImage;
    }
    else
    {
      try
      {
        mitk::ImageReadAccessor readAccess(convertedImage, convertedImage->GetVolumeData(0));
        const void* cPointer = readAccess.GetData();

        mitk::ImageWriteAccessor writeAccess(imageInNode);
        void* vPointer = writeAccess.GetData();

        memcpy(vPointer, cPointer, img->width() * img->height() * (imageDepth / 8));
      }
      catch(mitk::Exception& e)
      {
        MITK_ERROR << "Failed to copy QtCamera image to DataStorage due to " << e.what() << std::endl;
      }
    }

    // We tell the node that it is modified so the next rendering event
    // will redraw it. Triggering this does not in itself guarantee a re-rendering.
    imageInNode->GetVtkImageData()->Modified();
    node->Modified();

    infos[0].m_IsLate = this->IsLate(time, dataType->GetTimeStampInNanoSeconds());
    infos[0].m_LagInMilliseconds = this->GetLagInMilliseconds(time, dataType->GetTimeStampInNanoSeconds());
  }
  return infos;
}

} // end namespace
