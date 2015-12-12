/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtAudioDataSourceService.h"
#include "niftkQtAudioDataType.h"
#include <niftkIGIDataSourceI.h>
#include <mitkExceptionMacro.h>
#include <QDir>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker QtAudioDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
QtAudioDataSourceService::QtAudioDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("QtAudio-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_FrameId(0)
{
  mitkThrow() << "Not implemented yet. Volunteers .... please step forward!";

  this->SetStatus("Initialising");

  int defaultFramesPerSecond = 25;

  QString deviceName = this->GetName();
  m_ChannelNumber = (deviceName.remove(0, 8)).toInt(); // Should match string QtAudio- above

  this->StartCapturing();

  // Set the interval based on desired number of frames per second.
  // So, 25 fps = 40 milliseconds.
  // However: If system slows down (eg. saving images), then Qt will
  // drop clock ticks, so in effect, you will get less than this.
  int intervalInMilliseconds = 1000 / defaultFramesPerSecond;

  this->SetTimeStampTolerance(intervalInMilliseconds*1000000*1.5);
  this->SetShouldUpdate(true);
  this->SetProperties(properties);
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
QtAudioDataSourceService::~QtAudioDataSourceService()
{
  this->StopCapturing();
  s_Lock.RemoveSource(m_ChannelNumber);
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties QtAudioDataSourceService::GetProperties() const
{
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::StartCapturing()
{
  this->SetStatus("Capturing");
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::StopCapturing()
{
  this->SetStatus("Stopped");
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::CleanBuffer()
{
}


//-----------------------------------------------------------------------------
QString QtAudioDataSourceService::GetRecordingDirectoryName()
{
  return this->GetRecordingLocation()
      + this->GetPreferredSlash()
      + this->GetName()
      + "_" + (tr("%1").arg(m_ChannelNumber))
      ;
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataType::IGITimeType lastTimeStamp)
{
  QMutexLocker locker(&m_Lock);

  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);

  QDir directory(this->GetRecordingDirectoryName());
  if (directory.exists())
  {
    m_PlaybackIndex = ProbeTimeStampFiles(directory, QString(".png"));
  }
  else
  {
    assert(false);
  }
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::StopPlayback()
{
  QMutexLocker locker(&m_Lock);

  m_PlaybackIndex.clear();

  IGIDataSource::StopPlayback();
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp)
{
/*
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
        wrapper->SetIsSaved(false);

        // Buffer itself should be threadsafe, so I'm not locking anything here.
        m_Buffer->AddToBuffer(wrapper.GetPointer());

        cvReleaseImage(&img);
      }
    }
    this->SetStatus("Playing back");
  }
  */
}


//-----------------------------------------------------------------------------
bool QtAudioDataSourceService::ProbeRecordedData(const QString& path,
                                                     niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                                                     niftk::IGIDataType::IGITimeType* lastTimeStampInStore)
{
  // zero is a suitable default value. it's unlikely that anyone recorded a legitime data set in the middle ages.
  niftk::IGIDataType::IGITimeType  firstTimeStampFound = 0;
  niftk::IGIDataType::IGITimeType  lastTimeStampFound  = 0;

  /*
  // needs to match what SaveData() does below
  QDir directory(path);
  if (directory.exists())
  {
    std::set<niftk::IGIDataType::IGITimeType> timeStamps = ProbeTimeStampFiles(directory, QString(".jpg"));
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
  */
  return firstTimeStampFound != 0;
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::GrabData()
{
  {
    QMutexLocker locker(&m_Lock);

    if (this->GetIsPlayingBack())
    {
      return;
    }
  }

  /*
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
  wrapper->SetIsSaved(false);

  */

  //m_Buffer->AddToBuffer(wrapper.GetPointer());

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
void QtAudioDataSourceService::SaveItem(niftk::IGIDataType::Pointer data)
{
  /*
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
  */
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> QtAudioDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;

  // This loads playback-data into the buffers, so must
  // come before the check for empty buffer.
  if (this->GetIsPlayingBack())
  {
    this->PlaybackData(time);
  }
/*
  if (m_Buffer->GetBufferSize() == 0)
  {
    return infos;
  }

  if(m_Buffer->GetFirstTimeStamp() > time)
  {
    MITK_DEBUG << "QtAudioDataSourceService::Update(), requested time is before buffer time! "
               << " Buffer size=" << m_Buffer->GetBufferSize()
               << ", time=" << time
               << ", firstTime=" << m_Buffer->GetFirstTimeStamp();
    return infos;
  }
*/
  // Create default return status.
  IGIDataItemInfo info;
  info.m_Name = this->GetName();
//  info.m_FramesPerSecond = m_Buffer->GetFrameRate();
  infos.push_back(info);

  return infos;
}

} // end namespace
