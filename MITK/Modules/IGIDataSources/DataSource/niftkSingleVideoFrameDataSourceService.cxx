/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleVideoFrameDataSourceService.h"
#include <niftkIGIDataSourceUtils.h>
#include <mitkExceptionMacro.h>
#include <mitkImage.h>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <QDir>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker SingleVideoFrameDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
SingleVideoFrameDataSourceService::SingleVideoFrameDataSourceService(
  QString factoryName,
  const IGIDataSourceProperties& properties,
  mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("SingleVideoFrame-")
                 + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_FrameId(0)
, m_Buffer(nullptr)
, m_BackgroundDeleteThread(nullptr)
{
  this->SetStatus("Initialising");

  QString deviceName = this->GetName();
  m_ChannelNumber = (deviceName.remove(0, 17)).toInt(); // Should match string SingleVideoFrame- above

  int defaultFramesPerSecond = 25;
  m_Buffer = niftk::IGIDataSourceBuffer::New(defaultFramesPerSecond * 2);

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

  this->SetDescription("Local video source.");
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
SingleVideoFrameDataSourceService::~SingleVideoFrameDataSourceService()
{
  s_Lock.RemoveSource(m_ChannelNumber);

  m_BackgroundDeleteThread->ForciblyStop();
  delete m_BackgroundDeleteThread;
}


//-----------------------------------------------------------------------------
void SingleVideoFrameDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  if (properties.contains("lag"))
  {
    int milliseconds = (properties.value("lag")).toInt();
    m_Buffer->SetLagInMilliseconds(milliseconds);

    MITK_INFO << "SingleVideoFrameDataSourceService(" << this->GetName().toStdString()
              << "): Set lag to " << milliseconds << " ms.";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties SingleVideoFrameDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  props.insert("lag", m_Buffer->GetLagInMilliseconds());

  MITK_INFO << "SingleVideoFrameDataSourceService(:" << this->GetName().toStdString()
            << "):Retrieved current value of lag as " << m_Buffer->GetLagInMilliseconds();

  return props;
}


//-----------------------------------------------------------------------------
void SingleVideoFrameDataSourceService::CleanBuffer()
{
  // Buffer itself should be threadsafe.
  m_Buffer->CleanBuffer();
}


//-----------------------------------------------------------------------------
bool SingleVideoFrameDataSourceService::ProbeRecordedData(niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
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
void SingleVideoFrameDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
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
void SingleVideoFrameDataSourceService::StopPlayback()
{
  QMutexLocker locker(&m_Lock);

  m_PlaybackIndex.clear();
  m_Buffer->DestroyBuffer();

  IGIDataSource::StopPlayback();
}


//-----------------------------------------------------------------------------
void SingleVideoFrameDataSourceService::PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp)
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

      niftk::IGIDataType::Pointer wrapper = this->LoadImage(filename.str());
      if (wrapper.IsNull())
      {
        mitkThrow() << "Failed to load image:" << filename.str();
      }
      m_Buffer->AddToBuffer(wrapper.GetPointer());
    }
    this->SetStatus("Playing back");
  }
}


//-----------------------------------------------------------------------------
void SingleVideoFrameDataSourceService::GrabData()
{
  {
    QMutexLocker locker(&m_Lock);

    if (this->GetIsPlayingBack())
    {
      return;
    }
  }

  niftk::IGIDataType::Pointer wrapper = this->GrabImage();
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
void SingleVideoFrameDataSourceService::SaveItem(niftk::IGIDataType::Pointer data)
{
  QString directoryPath = this->GetRecordingDirectory();
  QDir directory(directoryPath);
  if (directory.mkpath(directoryPath))
  {
    QString fileName =  directoryPath + QDir::separator() + tr("%1.jpg").arg(data->GetTimeStampInNanoSeconds());
    this->SaveImage(fileName.toStdString(), data);
    data->SetIsSaved(true);
  }
  else
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as could not create " << directoryPath.toStdString();
  }
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> SingleVideoFrameDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
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
    MITK_DEBUG << "SingleVideoFrameDataSourceService::Update(), requested time is before buffer time! "
               << " Buffer size=" << m_Buffer->GetBufferSize()
               << ", time=" << time
               << ", firstTime=" << m_Buffer->GetFirstTimeStamp();
    return infos;
  }

  niftk::IGIDataType::Pointer dataType = m_Buffer->GetItem(time);
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

  unsigned int numberOfBytes = 0;
  mitk::Image::Pointer convertedImage = this->ConvertImage(dataType, numberOfBytes);
  if (numberOfBytes == 0)
  {
    mitkThrow() << "Failed to convert data (zero bytes!)";
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

      memcpy(vPointer, cPointer, numberOfBytes);
    }
    catch(mitk::Exception& e)
    {
      MITK_ERROR << "Failed to copy OpenCV image to DataStorage due to " << e.what() << std::endl;
    }
  }

  // We tell the node that it is modified so the next rendering event
  // will redraw it. Triggering this does not in itself guarantee a re-rendering.
  imageInNode->GetVtkImageData()->Modified();
  node->Modified();

  infos[0].m_IsLate = this->IsLate(time, dataType->GetTimeStampInNanoSeconds());
  infos[0].m_LagInMilliseconds = this->GetLagInMilliseconds(time, dataType->GetTimeStampInNanoSeconds());

  return infos;
}

} // end namespace
