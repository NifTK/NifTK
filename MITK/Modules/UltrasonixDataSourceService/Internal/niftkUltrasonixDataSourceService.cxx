/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUltrasonixDataSourceService.h"
#include "niftkUltrasonixDataType.h"
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
niftk::IGIDataSourceLocker UltrasonixDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
UltrasonixDataSourceService::UltrasonixDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("UltrasonixNetworked-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_FrameId(0)
, m_Buffer(NULL)
, m_BackgroundDeleteThread(NULL)
, m_DataGrabbingThread(NULL)
{
  mitkThrow() << "Not implemented yet. Volunteers .... please step forward!";

  this->SetStatus("Initialising");

  int defaultFramesPerSecond = 25;
  m_Buffer = niftk::IGIDataSourceBuffer::New(defaultFramesPerSecond * 2);

  QString deviceName = this->GetName();
  m_ChannelNumber = (deviceName.remove(0, 20)).toInt(); // Should match string UltrasonixNetworked- above

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

  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceService::~UltrasonixDataSourceService()
{
  s_Lock.RemoveSource(m_ChannelNumber);

  m_DataGrabbingThread->ForciblyStop();
  delete m_DataGrabbingThread;

  m_BackgroundDeleteThread->ForciblyStop();
  delete m_BackgroundDeleteThread;
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  if (properties.contains("lag"))
  {
    int milliseconds = (properties.value("lag")).toInt();
    m_Buffer->SetLagInMilliseconds(milliseconds);

    MITK_INFO << "UltrasonixDataSourceService(" << this->GetName().toStdString()
              << "): Set lag to " << milliseconds << " ms.";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties UltrasonixDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  props.insert("lag", m_Buffer->GetLagInMilliseconds());

  MITK_INFO << "UltrasonixDataSourceService(:" << this->GetName().toStdString()
            << "):Retrieved current value of lag as " << m_Buffer->GetLagInMilliseconds();

  return props;
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceService::CleanBuffer()
{
  // Buffer itself should be threadsafe.
  m_Buffer->CleanBuffer();
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataType::IGITimeType lastTimeStamp)
{
  /*
  QMutexLocker locker(&m_Lock);

  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);

  m_Buffer->DestroyBuffer();

  QDir directory(this->GetRecordingDirectoryName());
  if (directory.exists())
  {
    std::set<niftk::IGIDataType::IGITimeType> timeStamps;
    niftk::ProbeTimeStampFiles(directory, QString(".png"), timeStamps);
    m_PlaybackIndex = timeStamps;
  }
  else
  {
    assert(false);
  }
  */
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceService::StopPlayback()
{
  QMutexLocker locker(&m_Lock);

  m_PlaybackIndex.clear();
  m_Buffer->DestroyBuffer();

  IGIDataSource::StopPlayback();
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceService::PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp)
{
}


//-----------------------------------------------------------------------------
bool UltrasonixDataSourceService::ProbeRecordedData(niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                                                    niftk::IGIDataType::IGITimeType* lastTimeStampInStore)
{
  // zero is a suitable default value. it's unlikely that anyone recorded a legitime data set in the middle ages.
  niftk::IGIDataType::IGITimeType  firstTimeStampFound = 0;
  niftk::IGIDataType::IGITimeType  lastTimeStampFound  = 0;

  return firstTimeStampFound != 0;
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceService::GrabData()
{
  {
    QMutexLocker locker(&m_Lock);

    if (this->GetIsPlayingBack())
    {
      return;
    }
  }

  // Save synchronously.
  // This has the side effect that if saving is too slow,
  // the QTimers just won't keep up, and start missing pulses.
  if (this->GetIsRecording())
  {
    //this->SaveItem(wrapper.GetPointer());
  }
  this->SetStatus("Grabbing");
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceService::SaveItem(niftk::IGIDataType::Pointer data)
{
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> UltrasonixDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
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
    MITK_DEBUG << "UltrasonixDataSourceService::Update(), requested time is before buffer time! "
               << " Buffer size=" << m_Buffer->GetBufferSize()
               << ", time=" << time
               << ", firstTime=" << m_Buffer->GetFirstTimeStamp();
    return infos;
  }

  // Create default return status.
  IGIDataItemInfo info;
  info.m_Name = this->GetName();
  info.m_IsLate = false;
  info.m_FramesPerSecond = m_Buffer->GetFrameRate();
  info.m_LagInMilliseconds = 0;
  infos.push_back(info);
  return infos;
}

} // end namespace
