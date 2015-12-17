/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOpenCVVideoDataSourceService.h"
#include "niftkOpenCVVideoDataType.h"
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataSourceUtils.h>
#include <niftkImageConversion.h>
#include <mitkExceptionMacro.h>
#include <mitkImage.h>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <QDir>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker OpenCVVideoDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
OpenCVVideoDataSourceService::OpenCVVideoDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("OpenCV-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_FrameId(0)
, m_Buffer(NULL)
, m_BackgroundDeleteThread(NULL)
, m_DataGrabbingThread(NULL)
{
  this->SetStatus("Initialising");

  int defaultFramesPerSecond = 25;
  m_Buffer = niftk::IGIDataSourceBuffer::New(defaultFramesPerSecond * 2);

  QString deviceName = this->GetName();
  m_ChannelNumber = (deviceName.remove(0, 7)).toInt(); // Should match string OpenCV- above

  m_VideoSource = mitk::OpenCVVideoSource::New();
  m_VideoSource->SetVideoCameraInput(m_ChannelNumber);
  if (!m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StartCapturing();
  }

  // Check we can actually grab, as MITK class doesn't throw exceptions on creation.
  m_VideoSource->FetchFrame();
  const IplImage* img = m_VideoSource->GetCurrentFrame();
  if (img == NULL)
  {
    s_Lock.RemoveSource(m_ChannelNumber);
    mitkThrow() << "Failed to create " << this->GetName().toStdString()
                << ", please check log file!";
  }

  // Set the interval based on desired number of frames per second.
  // So, 25 fps = 40 milliseconds.
  // However: If system slows down (eg. saving images), then Qt will
  // drop clock ticks, so in effect, you will get less than this.
  int intervalInMilliseconds = 1000 / defaultFramesPerSecond;

  this->SetTimeStampTolerance(intervalInMilliseconds*1000000*1.5);
  this->SetShouldUpdate(true);
  this->SetProperties(properties);

  m_BackgroundDeleteThread = new niftk::IGIDataSourceBackgroundDeleteThread(NULL, this);
  m_BackgroundDeleteThread->SetInterval(2000); // try deleting images every 2 seconds.
  m_BackgroundDeleteThread->start();
  if (!m_BackgroundDeleteThread->isRunning())
  {
    mitkThrow() << "Failed to start background deleting thread";
  }

  m_DataGrabbingThread = new niftk::IGIDataSourceGrabbingThread(NULL, this);
  m_DataGrabbingThread->SetInterval(intervalInMilliseconds);
  m_DataGrabbingThread->start();
  if (!m_DataGrabbingThread->isRunning())
  {
    mitkThrow() << "Failed to start data grabbing thread";
  }

  this->SetDescription("Local video source, (via OpenCV), e.g. web-cam.");
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
OpenCVVideoDataSourceService::~OpenCVVideoDataSourceService()
{
  if (m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StopCapturing();
  }

  s_Lock.RemoveSource(m_ChannelNumber);

  m_DataGrabbingThread->ForciblyStop();
  delete m_DataGrabbingThread;

  m_BackgroundDeleteThread->ForciblyStop();
  delete m_BackgroundDeleteThread;
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  if (properties.contains("lag"))
  {
    int milliseconds = (properties.value("lag")).toInt();
    m_Buffer->SetLagInMilliseconds(milliseconds);

    MITK_INFO << "OpenCVVideoDataSourceService(" << this->GetName().toStdString()
              << "): Set lag to " << milliseconds << " ms.";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties OpenCVVideoDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  props.insert("lag", m_Buffer->GetLagInMilliseconds());

  MITK_INFO << "OpenCVVideoDataSourceService(:" << this->GetName().toStdString()
            << "):Retrieved current value of lag as " << m_Buffer->GetLagInMilliseconds();

  return props;
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::CleanBuffer()
{
  // Buffer itself should be threadsafe.
  m_Buffer->CleanBuffer();
}


//-----------------------------------------------------------------------------
QString OpenCVVideoDataSourceService::GetRecordingDirectoryName()
{
  return this->GetRecordingLocation()
      + niftk::GetPreferredSlash()
      + this->GetName()
      + "_" + (tr("%1").arg(m_ChannelNumber))
      ;
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataType::IGITimeType lastTimeStamp)
{
  QMutexLocker locker(&m_Lock);

  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);

  m_Buffer->DestroyBuffer();

  QDir directory(this->GetRecordingDirectoryName());
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
void OpenCVVideoDataSourceService::StopPlayback()
{
  QMutexLocker locker(&m_Lock);

  m_PlaybackIndex.clear();
  m_Buffer->DestroyBuffer();

  IGIDataSource::StopPlayback();
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp)
{
  assert(this->GetIsPlayingBack());
  assert(m_PlaybackIndex.size() > 0); // Should have failed probing if no data.

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
      filename << this->GetRecordingDirectoryName().toStdString() << '/' << (*i) << ".jpg";

      IplImage* img = cvLoadImage(filename.str().c_str());
      if (img)
      {
        niftk::OpenCVVideoDataType::Pointer wrapper = niftk::OpenCVVideoDataType::New();
        wrapper->CloneImage(img);
        wrapper->SetTimeStampInNanoSeconds(*i);
        wrapper->SetFrameId(m_FrameId++);
        wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
        wrapper->SetShouldBeSaved(false);

        // Buffer itself should be threadsafe, so I'm not locking anything here.
        m_Buffer->AddToBuffer(wrapper.GetPointer());

        cvReleaseImage(&img);
      }
    }
    this->SetStatus("Playing back");
  }
}


//-----------------------------------------------------------------------------
bool OpenCVVideoDataSourceService::ProbeRecordedData(const QString& path,
                                                     niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                                                     niftk::IGIDataType::IGITimeType* lastTimeStampInStore)
{
  // zero is a suitable default value. it's unlikely that anyone recorded a legitime data set in the middle ages.
  niftk::IGIDataType::IGITimeType  firstTimeStampFound = 0;
  niftk::IGIDataType::IGITimeType  lastTimeStampFound  = 0;

  // needs to match what SaveData() does below
  QDir directory(path);
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
void OpenCVVideoDataSourceService::GrabData()
{
  {
    QMutexLocker locker(&m_Lock);

    if (this->GetIsPlayingBack())
    {
      return;
    }
  }

  // Somehow this can become null, probably a race condition during destruction.
  if (m_VideoSource.IsNull())
  {
    mitkThrow() << "Video source is null. This should not happen! It's most likely a race-condition.";
  }

  // Grab a video image.
  m_VideoSource->FetchFrame();
  const IplImage* img = m_VideoSource->GetCurrentFrame();
  if (img == NULL)
  {
    mitkThrow() << "Failed to get a valid video frame!";
  }

  niftk::OpenCVVideoDataType::Pointer wrapper = niftk::OpenCVVideoDataType::New();
  wrapper->CloneImage(img);
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
  }

  // Putting this after the save, as we don't want to
  // add to the buffer in this grabbing thread, then the
  // m_BackgroundDeleteThread deletes the object while
  // we are trying to save the data.
  m_Buffer->AddToBuffer(wrapper.GetPointer());

  this->SetStatus("Grabbing");
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::SaveItem(niftk::IGIDataType::Pointer data)
{
  niftk::OpenCVVideoDataType::Pointer dataType = static_cast<niftk::OpenCVVideoDataType*>(data.GetPointer());
  if (dataType.IsNull())
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as the data received was the wrong type!";
  }

  const IplImage* imageFrame = dataType->GetImage();
  if (imageFrame == NULL)
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as the image frame was NULL!";
  }

  QString directoryPath = this->GetRecordingDirectoryName();
  QDir directory(directoryPath);
  if (directory.mkpath(directoryPath))
  {
    QString fileName =  directoryPath + QDir::separator() + tr("%1.jpg").arg(data->GetTimeStampInNanoSeconds());
    bool success = cvSaveImage(fileName.toStdString().c_str(), imageFrame);
    if (!success)
    {
      mitkThrow() << "Failed to save OpenCVVideoDataType in cvSaveImage!";
    }
    data->SetIsSaved(true);
  }
  else
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as could not create " << directoryPath.toStdString();
  }
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> OpenCVVideoDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;

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
    MITK_DEBUG << "OpenCVVideoDataSourceService::Update(), requested time is before buffer time! "
               << " Buffer size=" << m_Buffer->GetBufferSize()
               << ", time=" << time
               << ", firstTime=" << m_Buffer->GetFirstTimeStamp();
    return infos;
  }

  niftk::OpenCVVideoDataType::Pointer dataType = static_cast<niftk::OpenCVVideoDataType*>(m_Buffer->GetItem(time).GetPointer());
  if (dataType.IsNull())
  {
    MITK_DEBUG << "Failed to find data for time " << time << ", size=" << m_Buffer->GetBufferSize() << ", last=" << m_Buffer->GetLastTimeStamp() << std::endl;
    return infos;
  }

  // Create default return status.
  IGIDataItemInfo info;
  info.m_Name = this->GetName();
  info.m_FramesPerSecond = m_Buffer->GetFrameRate();
  infos.push_back(info);

  // If we are not actually updating data, bail out.
  if (!this->GetShouldUpdate())
  {
    return infos;
  }

  mitk::DataNode::Pointer node = this->GetDataNode(this->GetName());
  if (node.IsNull())
  {
    mitkThrow() << "Can't find mitk::DataNode with name " << this->GetName().toStdString() << std::endl;
  }

  // Get Image from the dataType;
  const IplImage* img = dataType->GetImage();
  if (img == NULL)
  {
    this->SetStatus("Failed");
    mitkThrow() << "Failed to extract OpenCV image from buffer!";
  }

  // OpenCV's cannonical channel layout is bgr (instead of rgb),
  // while everything usually else expects rgb...
  IplImage* rgbaOpenCVImage = cvCreateImage( cvSize( img->width, img->height ), img->depth, 4);
  cvCvtColor( img, rgbaOpenCVImage,  CV_BGR2RGBA );

  // ...so when we eventually extend/generalise CreateMitkImage() to handle different formats/etc
  // we should make sure we got the layout right. (opencv itself does not use this in any way.)
  std::memcpy(&rgbaOpenCVImage->channelSeq[0], "RGBA", 4);

  // And then we stuff it into the DataNode, where the SmartPointer will delete for us if necessary.
  mitk::Image::Pointer convertedImage = niftk::CreateMitkImage(rgbaOpenCVImage);

#ifdef XXX_USE_CUDA
  // a compatibility stop-gap to interface with new renderer and cuda bits.
  {
    CUDAManager*    cm = CUDAManager::GetInstance();
    if (cm != 0)
    {
      cudaStream_t    mystream = cm->GetStream("QmitkIGIOpenCVDataSource::Update");
      WriteAccessor   wa       = cm->RequestOutputImage(rgbaOpenCVImage->width, rgbaOpenCVImage->height, 4);

      assert(rgbaOpenCVImage->widthStep >= (rgbaOpenCVImage->width * 4));
      cudaMemcpy2DAsync(wa.m_DevicePointer, wa.m_BytePitch, rgbaOpenCVImage->imageData, rgbaOpenCVImage->widthStep, rgbaOpenCVImage->width * 4, rgbaOpenCVImage->height, cudaMemcpyHostToDevice, mystream);
      // no error handling...

      LightweightCUDAImage lwci = cm->Finalise(wa, mystream);

      CUDAImageProperty::Pointer    lwciprop = CUDAImageProperty::New();
      lwciprop->Set(lwci);

      convertedImage->SetProperty("CUDAImageProperty", lwciprop);
    }
  }
#endif

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

      memcpy(vPointer, cPointer, img->width * img->height * 4);
    }
    catch(mitk::Exception& e)
    {
      MITK_ERROR << "Failed to copy OpenCV image to DataStorage due to " << e.what() << std::endl;
    }
#ifdef _USE_CUDA
    imageInNode->SetProperty("CUDAImageProperty", convertedImage->GetProperty("CUDAImageProperty"));
#endif
  }

  // We tell the node that it is modified so the next rendering event
  // will redraw it. Triggering this does not in itself guarantee a re-rendering.
  imageInNode->GetVtkImageData()->Modified();
  node->Modified();

  cvReleaseImage(&rgbaOpenCVImage);

  infos[0].m_IsLate = this->IsLate(time, dataType->GetTimeStampInNanoSeconds());
  infos[0].m_LagInMilliseconds = this->GetLagInMilliseconds(time, dataType->GetTimeStampInNanoSeconds());
  return infos;
}

} // end namespace
