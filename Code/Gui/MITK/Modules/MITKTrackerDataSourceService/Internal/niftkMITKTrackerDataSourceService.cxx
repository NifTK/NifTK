/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKTrackerDataSourceService.h"
#include <niftkIGITrackerDataType.h>
#include <mitkExceptionMacro.h>
#include <QDir>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
QMutex    MITKTrackerDataSourceService::s_Lock(QMutex::Recursive);
QSet<int> MITKTrackerDataSourceService::s_SourcesInUse;


//-----------------------------------------------------------------------------
int MITKTrackerDataSourceService::GetNextTrackerNumber()
{
  s_Lock.lock();
  unsigned int sourceCounter = 0;
  while(s_SourcesInUse.contains(sourceCounter))
  {
    sourceCounter++;
  }
  s_SourcesInUse.insert(sourceCounter);
  s_Lock.unlock();
  return sourceCounter;
}


//-----------------------------------------------------------------------------
MITKTrackerDataSourceService::MITKTrackerDataSourceService(
    QString name,
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage,
    niftk::NDITracker::Pointer tracker
    )
: IGIDataSource((name + QString("-") + QString::number(GetNextTrackerNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_FrameId(0)
, m_BackgroundDeleteThread(NULL)
, m_DataGrabbingThread(NULL)
, m_Lag(0)
, m_Tracker(tracker)
{
  if (m_Tracker.IsNull())
  {
    mitkThrow() << "Tracker is NULL!";
  }

  this->SetStatus("Initialising");

  // Trigger an update, just in case it crashes.
  // Its best to bail out during constructor.
  m_Tracker->StartTracking();
  m_Tracker->Update();

  QString deviceName = this->GetName();
  m_TrackerNumber = (deviceName.remove(0, name.length() + 1)).toInt();

  m_BackgroundDeleteThread = new niftk::IGIDataSourceBackgroundDeleteThread(NULL, this);
  m_BackgroundDeleteThread->SetInterval(1000); // try deleting data every 1 second.
  m_BackgroundDeleteThread->start();
  if (!m_BackgroundDeleteThread->isRunning())
  {
    mitkThrow() << "Failed to start background deleting thread";
  }

  // Set the interval based on desired number of frames per second.
  // eg. 25 fps = 40 milliseconds.
  // However: If system slows down (eg. saving images), then Qt will
  // drop clock ticks, so in effect, you will get less than this.
  int defaultFramesPerSecond = m_Tracker->GetPreferredFramesPerSecond();
  int intervalInMilliseconds = 1000 / defaultFramesPerSecond;

  m_DataGrabbingThread = new niftk::IGIDataSourceGrabbingThread(NULL, this);
  m_DataGrabbingThread->SetInterval(intervalInMilliseconds);
  m_DataGrabbingThread->start();
  if (!m_DataGrabbingThread->isRunning())
  {
    mitkThrow() << "Failed to start data grabbing thread";
  }

  this->SetTimeStampTolerance(intervalInMilliseconds*1000000*1.1);
  this->SetProperties(properties);
  this->SetShouldUpdate(true);
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
MITKTrackerDataSourceService::~MITKTrackerDataSourceService()
{
  this->StopCapturing();

  s_Lock.lock();
  s_SourcesInUse.remove(m_TrackerNumber);
  s_Lock.unlock();

  m_DataGrabbingThread->ForciblyStop();
  delete m_DataGrabbingThread;

  m_BackgroundDeleteThread->ForciblyStop();
  delete m_BackgroundDeleteThread;
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  // In contrast say, to the OpenCV source, we don't set the lag
  // directly on the buffer because, there may be no buffers present
  // at the time this method is called.
  if (properties.contains("lag"))
  {
    int milliseconds = (properties.value("lag")).toInt();
    m_Lag = milliseconds;

    MITK_INFO << "MITKTrackerDataSourceService(" << this->GetName().toStdString()
              << "): set lag to " << milliseconds << " ms.";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties MITKTrackerDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  props.insert("lag", m_Lag);

  MITK_INFO << "MITKTrackerDataSourceService:(" << this->GetName().toStdString()
            << "): Retrieved current value of lag as " << m_Lag << " ms.";

  return props;
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::StartCapturing()
{
  m_Tracker->StartTracking();
  this->SetStatus("Capturing");
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::StopCapturing()
{
  m_Tracker->StopTracking();
  this->SetStatus("Stopped");
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::CleanBuffer()
{
  // Buffer itself should be threadsafe. Clean all buffers.
  QMap<QString, niftk::IGIDataSourceBuffer::Pointer>::iterator iter;
  for (iter = m_Buffers.begin(); iter != m_Buffers.end(); iter++)
  {
    (*iter)->CleanBuffer();
  }
}


//-----------------------------------------------------------------------------
QString MITKTrackerDataSourceService::GetRecordingDirectoryName()
{
  return this->GetRecordingLocation()
      + this->GetPreferredSlash()
      + this->GetName()
      + "_"
      ;
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataType::IGITimeType lastTimeStamp)
{
  QMutexLocker locker(&m_Lock);

  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);

  m_Buffers.clear();

  QDir recordingDir(this->GetRecordingDirectoryName());
  if (recordingDir.exists())
  {
    // then directories with tool names
    recordingDir.setFilter(QDir::Dirs | QDir::Readable | QDir::NoDotAndDotDot);

    QStringList toolNames = recordingDir.entryList();
    if (!toolNames.isEmpty())
    {
      foreach (QString tool, toolNames)
      {
        QDir  tooldir(recordingDir.path() + QDir::separator() + tool);
        assert(tooldir.exists());

        std::set<niftk::IGIDataType::IGITimeType> timeStamps = ProbeTimeStampFiles(tooldir, QString(".txt"));
        if (!timeStamps.empty())
        {
          m_PlaybackIndex.insert(tool, timeStamps);
        }
      }
    }
    else
    {
      MITK_WARN << "There are no tool sub-folders in " << recordingDir.absolutePath().toStdString() << ", so can't playback tracking data!";
      return;
    }
  }
  else
  {
    mitkThrow() << this->GetName().toStdString() << ": Recording directory, " << recordingDir.absolutePath().toStdString() << ", does not exist!";
  }
  if (m_PlaybackIndex.size())
  {
    mitkThrow() << "No tracking data extracted from directory " << recordingDir.absolutePath().toStdString();
  }
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::StopPlayback()
{
  QMutexLocker locker(&m_Lock);

  m_PlaybackIndex.clear();
  m_Buffers.clear();

  IGIDataSource::StopPlayback();
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp)
{
  assert(this->GetIsPlayingBack());
  assert(m_PlaybackIndex.size() > 0); // Should have failed probing if no data.

  // This will find us the timestamp right after the requested one.
  // Remember we have multiple buffers!
  QMap<QString, niftk::IGIDataSourceBuffer::Pointer>::iterator iter;
  for (iter = m_Buffers.begin(); iter != m_Buffers.end(); iter++)
  {
    QString bufferName = iter.key();
    if (!m_PlaybackIndex.contains(bufferName))
    {
      mitkThrow() << "Invalid buffer name found " << bufferName.toStdString();
    }

    std::set<niftk::IGIDataType::IGITimeType>::const_iterator i = m_PlaybackIndex[bufferName].upper_bound(requestedTimeStamp);
    if (i != m_PlaybackIndex[bufferName].begin())
    {
      --i;
    }
    if (i != m_PlaybackIndex[bufferName].end())
    {
      if (!m_Buffers[bufferName]->Contains(*i))
      {
        std::ostringstream  filename;
        filename << this->GetRecordingDirectoryName().toStdString()
                 << this->GetPreferredSlash().toStdString()
                 << bufferName.toStdString()
                 << this->GetPreferredSlash().toStdString()
                 << (*i)
                 << ".txt";

        std::ifstream   file(filename.str().c_str());
        if (file)
        {
          vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
          matrix->Identity();

          for (int r = 0; r < 4; ++r)
          {
            for (int c = 0; c < 4; ++c)
            {
              double tmp;
              file >> tmp;
              matrix->SetElement(r,c,tmp);
            }
          }

          niftk::IGITrackerDataType::Pointer wrapper = niftk::IGITrackerDataType::New();
          wrapper->SetTimeStampInNanoSeconds(*i);
          wrapper->SetTrackingData(matrix);
          wrapper->SetFrameId(m_FrameId++);
          wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
          wrapper->SetShouldBeSaved(false);
          wrapper->SetIsSaved(false);

          // Buffer itself should be threadsafe, so I'm not locking anything here.
          m_Buffers[bufferName]->AddToBuffer(wrapper.GetPointer());

        } // end if file open
      } // end if item not already in buffer
    } // end: if we found a valid item to playback
  } // end: foreach buffer

  this->SetStatus("Playing back");
}


//-----------------------------------------------------------------------------
bool MITKTrackerDataSourceService::ProbeRecordedData(const QString& path,
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
void MITKTrackerDataSourceService::GrabData()
{
/*
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
  wrapper->SetIsSaved(false);

  m_Buffer->AddToBuffer(wrapper.GetPointer());

  // Save synchronously.
  // This has the side effect that if saving is too slow,
  // the QTimers just won't keep up, and start missing pulses.
  if (this->GetIsRecording())
  {
    this->SaveItem(wrapper.GetPointer());
  }
*/
  this->SetStatus("Grabbing");
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::SaveItem(niftk::IGIDataType::Pointer data)
{
/*
  niftk::OpenCVVideoDataType::Pointer dataType = static_cast<niftk::OpenCVVideoDataType*>(data.GetPointer());
  if (dataType.IsNull())
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as the data received was NULL!";
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
std::vector<IGIDataItemInfo> MITKTrackerDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;
/*
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
    MITK_DEBUG << "MITKTrackerDataSourceService::Update(), requested time is before buffer time! "
               << " Buffer size=" << m_Buffer->GetBufferSize()
               << ", time=" << time
               << ", firstTime=" << m_Buffer->GetFirstTimeStamp();
    return infos;
  }

  infos[0].m_IsLate = this->IsLate(time, dataType->GetTimeStampInNanoSeconds());
  infos[0].m_LagInMilliseconds = this->GetLagInMilliseconds(time, dataType->GetTimeStampInNanoSeconds());
*/
  return infos;
}

} // end namespace
