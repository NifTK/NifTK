/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKTrackerDataSourceService.h"
#include "niftkIGITrackerDataType.h"
#include <mitkCoordinateAxesData.h>
#include <mitkFileIOUtils.h>
#include <mitkExceptionMacro.h>
#include <QDir>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker MITKTrackerDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
MITKTrackerDataSourceService::MITKTrackerDataSourceService(
    QString name,
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage,
    niftk::NDITracker::Pointer tracker
    )
: IGIDataSource((name + QString("-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
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

  // Set the interval based on desired number of frames per second.
  // eg. 25 fps = 40 milliseconds.
  // However: If system slows down (eg. saving images), then Qt will
  // drop clock ticks, so in effect, you will get less than this.
  int defaultFramesPerSecond = m_Tracker->GetPreferredFramesPerSecond();
  int intervalInMilliseconds = 1000 / defaultFramesPerSecond;

  this->SetTimeStampTolerance(intervalInMilliseconds*1000000*1.1);
  this->SetProperties(properties);
  this->SetShouldUpdate(true);

  m_BackgroundDeleteThread = new niftk::IGIDataSourceBackgroundDeleteThread(NULL, this);
  m_BackgroundDeleteThread->SetInterval(1000); // try deleting data every 1 second.
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
MITKTrackerDataSourceService::~MITKTrackerDataSourceService()
{
  this->StopCapturing();

  s_Lock.RemoveSource(m_TrackerNumber);

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
  // at the time this method is called. For example, you could
  // have created a tracker, and no tracked objects are placed within
  // the field of view, thereby no tracking matrices would be generated.
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
      ;
}


//-----------------------------------------------------------------------------
QMap<QString, std::set<niftk::IGIDataType::IGITimeType> >  MITKTrackerDataSourceService::GetPlaybackIndex(QString directory)
{

  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> > result;

  QDir recordingDir(directory);
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
          result.insert(tool, timeStamps);
        }
      }
    }
    else
    {
      MITK_WARN << "There are no tool sub-folders in " << recordingDir.absolutePath().toStdString() << ", so can't playback tracking data!";
      return result;
    }
  }
  else
  {
    mitkThrow() << this->GetName().toStdString() << ": Recording directory, " << recordingDir.absolutePath().toStdString() << ", does not exist!";
  }
  if (result.isEmpty())
  {
    mitkThrow() << "No tracking data extracted from directory " << recordingDir.absolutePath().toStdString();
  }
  return result;
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataType::IGITimeType lastTimeStamp)
{
  QMutexLocker locker(&m_Lock);

  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);

  m_Buffers.clear();
  m_PlaybackIndex = this->GetPlaybackIndex(this->GetRecordingDirectoryName());
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

  niftk::IGIDataType::IGITimeType  firstTimeStampFound = std::numeric_limits<niftk::IGIDataType::IGITimeType>::max();
  niftk::IGIDataType::IGITimeType  lastTimeStampFound  = std::numeric_limits<niftk::IGIDataType::IGITimeType>::min();

  // Note, that each tool may have different min and max, so we want the
  // most minimum and most maximum of all the sub directories.

  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> > result = this->GetPlaybackIndex(path);
  if (result.isEmpty())
  {
    return false;
  }

  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> >::iterator iter;
  for (iter = result.begin(); iter != result.end(); iter++)
  {
    if (!iter.value().empty())
    {
      niftk::IGIDataType::IGITimeType first = *((*iter).begin());
      if (first < firstTimeStampFound)
      {
        firstTimeStampFound = first;
      }

      niftk::IGIDataType::IGITimeType last = *(--((*iter).end()));
      if (last > lastTimeStampFound)
      {
        lastTimeStampFound = last;
      }
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
  return firstTimeStampFound != std::numeric_limits<niftk::IGIDataType::IGITimeType>::max();
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::GrabData()
{
  {
    QMutexLocker locker(&m_Lock);

    if (this->GetIsPlayingBack())
    {
      return;
    }
  }
  if (m_Tracker.IsNull())
  {
    mitkThrow() << "Tracker is null. This should not happen! It's a programming bug.";
  }

  m_Tracker->Update();
  niftk::IGIDataType::IGITimeType timeCreated = this->GetTimeStampInNanoseconds();

  std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > result = m_Tracker->GetTrackingData();

  if (!result.empty())
  {
    std::map<std::string, vtkSmartPointer<vtkMatrix4x4> >::iterator iter;
    for (iter = result.begin(); iter != result.end(); iter++)
    {
      std::string toolName = (*iter).first;
      QString toolNameAsQString = QString::fromStdString(toolName);

      niftk::IGITrackerDataType::Pointer wrapper = niftk::IGITrackerDataType::New();
      wrapper->SetToolName(toolName);
      wrapper->SetTrackingData((*iter).second);
      wrapper->SetTimeStampInNanoSeconds(timeCreated);
      wrapper->SetFrameId(m_FrameId++);
      wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
      wrapper->SetShouldBeSaved(this->GetIsRecording());
      wrapper->SetIsSaved(false);

      if (!m_Buffers.contains(toolNameAsQString))
      {
        niftk::IGIDataSourceBuffer::Pointer newBuffer = niftk::IGIDataSourceBuffer::New(m_Tracker->GetPreferredFramesPerSecond() * 2);
        m_Buffers.insert(toolNameAsQString, newBuffer);
      }

      m_Buffers[toolNameAsQString]->AddToBuffer(wrapper.GetPointer());

      // Save synchronously.
      // This has the side effect that if saving is too slow,
      // the QTimers just won't keep up, and start missing pulses.
      if (this->GetIsRecording())
      {
        this->SaveItem(wrapper.GetPointer());
      }
    }
    this->SetStatus("Grabbing");
  }
  else
  {
    this->SetStatus("No data!");
  }
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::SaveItem(niftk::IGIDataType::Pointer data)
{

  niftk::IGITrackerDataType::Pointer dataType = static_cast<niftk::IGITrackerDataType*>(data.GetPointer());
  if (dataType.IsNull())
  {
    mitkThrow() << "Failed to save IGITrackerDataType as the data received was the wrong type!";
  }

  vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
  matrix = dataType->GetTrackingData();

  if (matrix == NULL)
  {
    mitkThrow() << "Failed to save IGITrackerDataType as the tracking matrix was NULL!";
  }

  QString directoryPath = this->GetRecordingDirectoryName();
  QString toolPath = directoryPath
      + this->GetPreferredSlash()
      + QString::fromStdString(dataType->GetToolName())
      + this->GetPreferredSlash();

  QDir directory(toolPath);
  if (directory.mkpath(toolPath))
  {
    QString fileName =  directoryPath + QDir::separator() + tr("%1.txt").arg(data->GetTimeStampInNanoSeconds());

    bool success = mitk::SaveVtkMatrix4x4ToFile(fileName.toStdString(), *matrix);
    if (!success)
    {
      mitkThrow() << "Failed to save IGITrackerDataType to " << fileName.toStdString();
    }
    data->SetIsSaved(true);
  }
  else
  {
    mitkThrow() << "Failed to save IGITrackerDataType as could not create " << directoryPath.toStdString();
  }
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> MITKTrackerDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;

  // This loads playback-data into the buffers, so must
  // come before the check for empty buffer.
  if (this->GetIsPlayingBack())
  {
    this->PlaybackData(time);
  }

  // Early exit if no buffers, which means that
  // the tracker is created, but has not seen anything to track yet.
  if (m_Buffers.isEmpty())
  {
    return infos;
  }

  if (!this->GetShouldUpdate())
  {
    return infos;
  }

  QMap<QString, niftk::IGIDataSourceBuffer::Pointer>::iterator iter;
  for (iter = m_Buffers.begin(); iter != m_Buffers.end(); iter++)
  {
    QString bufferName = iter.key();

    if (m_Buffers[bufferName]->GetBufferSize() == 0)
    {
      continue;
    }

    if(m_Buffers[bufferName]->GetFirstTimeStamp() > time)
    {
      continue;
    }

    niftk::IGITrackerDataType::Pointer dataType = static_cast<niftk::IGITrackerDataType*>(m_Buffers[bufferName]->GetItem(time).GetPointer());
    if (dataType.IsNull())
    {
      MITK_DEBUG << "Failed to find data for time " << time << ", size=" << m_Buffers[bufferName]->GetBufferSize() << ", last=" << m_Buffers[bufferName]->GetLastTimeStamp() << std::endl;
      continue;
    }

    mitk::DataNode::Pointer node = this->GetDataNode(bufferName);
    if (node.IsNull())
    {
      mitkThrow() << "Can't find mitk::DataNode with name " << bufferName.toStdString();
    }

    mitk::CoordinateAxesData::Pointer coords = static_cast<mitk::CoordinateAxesData*>(node->GetData());
    if (coords.IsNull())
    {
      mitkThrow() << "DataNode with name " << bufferName.toStdString() << " contains the wrong data type!";
    }

    vtkSmartPointer<vtkMatrix4x4> matrix = dataType->GetTrackingData();
    coords->SetVtkMatrix(*matrix);

    // We tell the node that it is modified so the next rendering event
    // will redraw it. Triggering this does not in itself guarantee a re-rendering.
    coords->Modified();
    node->Modified();

    IGIDataItemInfo info;
    info.m_Name = this->GetName();
    info.m_Status = this->GetStatus();
    info.m_ShouldUpdate = this->GetShouldUpdate();
    info.m_FramesPerSecond = m_Buffers[bufferName]->GetFrameRate();
    info.m_Description = "MITK based tracker.";
    info.m_IsLate = this->IsLate(time, dataType->GetTimeStampInNanoSeconds());
    info.m_LagInMilliseconds = this->GetLagInMilliseconds(time, dataType->GetTimeStampInNanoSeconds());
    infos.push_back(info);
  }

  return infos;
}

} // end namespace
