/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleFrameDataSourceService.h"
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
niftk::IGIDataSourceLocker SingleFrameDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
SingleFrameDataSourceService::SingleFrameDataSourceService(
  QString deviceName,
  QString factoryName,
  unsigned int framesPerSecond,
  unsigned int bufferSize,
  const IGIDataSourceProperties& properties,
  mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((deviceName + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_ChannelNumber(0)
, m_FrameId(0)
, m_Buffer(bufferSize)
, m_ApproxIntervalInMilliseconds(0)
, m_FileExtension(".jpg") // faster than .png, but lossy.
{
  this->SetStatus("Initialising");

  if(!properties.contains("extension"))
  {
    mitkThrow() << "File extension not specified!";
  }
  m_FileExtension = (properties.value("extension")).toString();

  QString fullDeviceName = this->GetName();
  m_ChannelNumber = (fullDeviceName.remove(0, deviceName.length())).toInt();

  this->SetApproximateIntervalInMilliseconds(1000 / framesPerSecond);
  this->SetShouldUpdate(true);
  this->SetProperties(properties);
  this->SetDescription("Local image source.");
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
SingleFrameDataSourceService::~SingleFrameDataSourceService()
{
  s_Lock.RemoveSource(m_ChannelNumber);
}


//-----------------------------------------------------------------------------
void SingleFrameDataSourceService::SetApproximateIntervalInMilliseconds(const int& ms)
{
  m_ApproxIntervalInMilliseconds = ms;

  // multiply by 1000000 to get nanoseconds
  // multiply by 5 so we start warning if timestamps suggest we are more than 5 frame intervals late.
  this->SetTimeStampTolerance(m_ApproxIntervalInMilliseconds*1000000*5);
}


//-----------------------------------------------------------------------------
void SingleFrameDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  if (properties.contains("lag"))
  {
    int milliseconds = (properties.value("lag")).toInt();
    m_Buffer.SetLagInMilliseconds(milliseconds);

    MITK_INFO << "SingleFrameDataSourceService(" << this->GetName().toStdString()
              << "): Set lag to " << milliseconds << " ms.";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties SingleFrameDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  props.insert("lag", m_Buffer.GetLagInMilliseconds());

  MITK_INFO << "SingleFrameDataSourceService(:" << this->GetName().toStdString()
            << "):Retrieved current value of lag as " << m_Buffer.GetLagInMilliseconds();

  return props;
}


//-----------------------------------------------------------------------------
bool SingleFrameDataSourceService::ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                                     niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  // zero is a suitable default value. it's unlikely that anyone recorded a legitime data set in the middle ages.
  niftk::IGIDataSourceI::IGITimeType firstTimeStampFound = 0;
  niftk::IGIDataSourceI::IGITimeType lastTimeStampFound  = 0;

  QDir directory(this->GetPlaybackDirectory());
  if (directory.exists())
  {
    std::set<niftk::IGIDataSourceI::IGITimeType> timeStamps;
    niftk::ProbeTimeStampFiles(directory, m_FileExtension, timeStamps);
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
void SingleFrameDataSourceService::StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataSourceI::IGITimeType lastTimeStamp)
{
  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);

  m_Buffer.CleanBuffer();

  QDir directory(this->GetPlaybackDirectory());
  if (directory.exists())
  {
    std::set<niftk::IGIDataSourceI::IGITimeType> timeStamps;
    niftk::ProbeTimeStampFiles(directory, m_FileExtension, timeStamps);
    m_PlaybackIndex = timeStamps;
  }
  else
  {
    assert(false);
  }
}


//-----------------------------------------------------------------------------
void SingleFrameDataSourceService::StopPlayback()
{
  m_PlaybackIndex.clear();
  m_Buffer.CleanBuffer();

  IGIDataSource::StopPlayback();
}


//-----------------------------------------------------------------------------
void SingleFrameDataSourceService::PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp)
{
  assert(this->GetIsPlayingBack());
  assert(!m_PlaybackIndex.empty()); // Should have failed probing if no data.

  // this will find us the timestamp right after the requested one
  std::set<niftk::IGIDataSourceI::IGITimeType>::const_iterator i = m_PlaybackIndex.upper_bound(requestedTimeStamp);
  if (i != m_PlaybackIndex.begin())
  {
    --i;
  }
  if (i != m_PlaybackIndex.end())
  {
    if (!m_Buffer.Contains(*i))
    {
      std::ostringstream  filename;
      filename << this->GetPlaybackDirectory().toStdString() << '/' << (*i) << m_FileExtension.toStdString();

      std::unique_ptr<niftk::IGIDataType> wrapper = this->LoadImage(filename.str());
      if (!wrapper)
      {
        mitkThrow() << "Failed to create wrapper for:" << filename.str();
      }
      wrapper->SetTimeStampInNanoSeconds(*i);
      wrapper->SetFrameId(m_FrameId++);
      wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
      wrapper->SetShouldBeSaved(false);
      m_Buffer.AddToBuffer(wrapper);
    }
    this->SetStatus("Playing back");
  }
}


//-----------------------------------------------------------------------------
void SingleFrameDataSourceService::GrabData()
{

  if (this->GetIsPlayingBack())
  {
    return;
  }

  std::unique_ptr<niftk::IGIDataType> wrapper = this->GrabImage();
  wrapper->SetTimeStampInNanoSeconds(this->GetTimeStampInNanoseconds());
  wrapper->SetFrameId(m_FrameId++);
  wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
  wrapper->SetShouldBeSaved(this->GetIsRecording());

  if (this->GetIsRecording())
  {
    this->SaveItem(*wrapper);
    this->SetStatus("Saving");
  }
  else
  {
    this->SetStatus("Grabbing");
  }

  m_Buffer.AddToBuffer(wrapper);
}


//-----------------------------------------------------------------------------
void SingleFrameDataSourceService::SaveItem(niftk::IGIDataType& data)
{
  QString directoryPath = this->GetRecordingDirectory();
  QDir directory(directoryPath);
  if (directory.mkpath(directoryPath))
  {
    QString fileName =  directoryPath + QDir::separator()
                        + tr("%1").arg(data.GetTimeStampInNanoSeconds()) + m_FileExtension;
    this->SaveImage(fileName.toStdString(), data);
    data.SetIsSaved(true);
  }
  else
  {
    mitkThrow() << "Failed to save image as could not create " << directoryPath.toStdString();
  }
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> SingleFrameDataSourceService::Update(const niftk::IGIDataSourceI::IGITimeType& time)
{
  m_Buffer.UpdateFrameRate();

  // Create default return status.
  // So, we always return at least 1 row.
  std::vector<IGIDataItemInfo> infos;
  IGIDataItemInfo info;
  info.m_Name = this->GetName();
  info.m_FramesPerSecond = m_Buffer.GetFrameRate();
  info.m_IsLate = true;
  info.m_LagInMilliseconds = 0;
  infos.push_back(info);

  // If we are not actually updating data, bail out.
  if (!this->GetShouldUpdate())
  {
    return infos;
  }

  // This loads playback-data into the buffers, so must
  // come before the check for empty buffer.
  if (this->GetIsPlayingBack())
  {
    this->PlaybackData(time);
  }

  if (m_Buffer.GetBufferSize() == 0)
  {
    return infos;
  }

  if(m_Buffer.GetFirstTimeStamp() > time)
  {
    MITK_DEBUG << "SingleFrameDataSourceService::Update(), requested time is before buffer time! "
               << " Buffer size=" << m_Buffer.GetBufferSize()
               << ", time=" << time
               << ", firstTime=" << m_Buffer.GetFirstTimeStamp();
    return infos;
  }

  mitk::DataNode::Pointer node = this->GetDataNode(this->GetName());
  if (node.IsNull())
  {
    mitkThrow() << "Can't find mitk::DataNode with name "
                << this->GetName().toStdString() << std::endl;
  }

  niftk::IGIDataSourceI::IGITimeType actualTime;
  unsigned int numberOfBytes = 0;

  mitk::Image::Pointer convertedImage = this->RetrieveImage(time, actualTime, numberOfBytes);
  if (numberOfBytes == 0)
  {
    MITK_DEBUG << "Failed to find data for time " << time
               << ", size=" << m_Buffer.GetBufferSize()
               << ", last=" << m_Buffer.GetLastTimeStamp() << std::endl;
    return infos;
  }

  mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());
  if (!imageInNode.IsNull())
  {
    // check size of image that is already attached to data node!
    bool haswrongsize = false;
    haswrongsize |= imageInNode->GetDimension(0) != convertedImage->GetDimension(0);
    haswrongsize |= imageInNode->GetDimension(1) != convertedImage->GetDimension(1);
    haswrongsize |= imageInNode->GetDimension(2) != 1;
    // check image type as well.
    haswrongsize |= imageInNode->GetPixelType().GetBitsPerComponent()
      != convertedImage->GetPixelType().GetBitsPerComponent();
    haswrongsize |= imageInNode->GetPixelType().GetNumberOfComponents()
      != convertedImage->GetPixelType().GetNumberOfComponents();

    if (haswrongsize)
    {
      imageInNode = mitk::Image::Pointer();
    }
  }

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
      MITK_ERROR << "Failed to copy image to DataStorage due to " << e.what() << std::endl;
    }
  }

#ifdef _USE_CUDA
  if (convertedImage->GetProperty("CUDAImageProperty").IsNotNull())
  {
    imageInNode->SetProperty("CUDAImageProperty", convertedImage->GetProperty("CUDAImageProperty"));
  }
#endif

  // We tell the node that it is modified so the next rendering event
  // will redraw it. Triggering this does not in itself guarantee a re-rendering.
  imageInNode->GetVtkImageData()->Modified();
  node->Modified();

  infos[0].m_IsLate = this->IsLate(time, actualTime);
  infos[0].m_LagInMilliseconds = this->GetLagInMilliseconds(time, actualTime);

  return infos;
}

} // end namespace
