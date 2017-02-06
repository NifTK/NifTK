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
#include <niftkIGIDataSourceUtils.h>
#include <niftkCoordinateAxesData.h>
#include <niftkFileIOUtils.h>

#include <mitkExceptionMacro.h>

#include <QDir>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceLocker MITKTrackerDataSourceService::s_Lock;

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
, m_Lag(0)
, m_FrameId(0)
, m_DataGrabbingThread(NULL)
, m_Tracker(tracker)
{
  if (m_Tracker.IsNull())
  {
    mitkThrow() << "Tracker is NULL!";
  }

  this->SetStatus("Initialising");

  QString deviceName = this->GetName();
  m_TrackerNumber = (deviceName.remove(0, name.length() + 1)).toInt();

  // Set the interval based on desired number of frames per second.
  // eg. 25 fps = 40 milliseconds.
  int defaultFramesPerSecond = m_Tracker->GetPreferredFramesPerSecond();
  int intervalInMilliseconds = 1000 / defaultFramesPerSecond;

  this->SetTimeStampTolerance(intervalInMilliseconds*1000000*5);
  this->SetProperties(properties);
  this->SetShouldUpdate(true);

  m_DataGrabbingThread = new niftk::IGIDataSourceGrabbingThread(NULL, this);
  m_DataGrabbingThread->SetInterval(intervalInMilliseconds);
  m_DataGrabbingThread->start();
  if (!m_DataGrabbingThread->isRunning())
  {
    mitkThrow() << "Failed to start data grabbing thread";
  }

  this->SetDescription("MITK Tracker:" + this->GetName());
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
MITKTrackerDataSourceService::~MITKTrackerDataSourceService()
{
  m_DataGrabbingThread->ForciblyStop();
  delete m_DataGrabbingThread;

  s_Lock.RemoveSource(m_TrackerNumber);
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
QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> >
MITKTrackerDataSourceService::GetPlaybackIndex(QString directory)
{
  QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> > bufferToTimeStamp;
  QMap<QString, QHash<niftk::IGIDataSourceI::IGITimeType, QStringList> > bufferToTimeStampToFileNames;

  niftk::GetPlaybackIndex(directory, QString(".txt"), bufferToTimeStamp, bufferToTimeStampToFileNames);
  return bufferToTimeStamp;
}


//-----------------------------------------------------------------------------
bool MITKTrackerDataSourceService::ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                                     niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  QString path = this->GetPlaybackDirectory();
  return niftk::ProbeRecordedData(path, QString(".txt"), firstTimeStampInStore, lastTimeStampInStore);
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataSourceI::IGITimeType lastTimeStamp)
{
  QMutexLocker locker(&m_Lock);

  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);

  m_Buffers.clear();
  m_PlaybackIndex = this->GetPlaybackIndex(this->GetPlaybackDirectory());
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
void MITKTrackerDataSourceService::PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp)
{
  assert(this->GetIsPlayingBack());
  assert(m_PlaybackIndex.size() > 0); // Should have failed probing if no data.

  // This will find us the timestamp right after the requested one.
  // Remember we have multiple buffers!
  QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> >::iterator playbackIter;
  for(playbackIter = m_PlaybackIndex.begin(); playbackIter != m_PlaybackIndex.end(); ++playbackIter)
  {
    QString bufferName = playbackIter.key();
    std::string bufferNameAsStdString = bufferName.toStdString();

    std::set<niftk::IGIDataSourceI::IGITimeType>::const_iterator i =
      m_PlaybackIndex[bufferName].upper_bound(requestedTimeStamp);

    if (i != m_PlaybackIndex[bufferName].begin())
    {
      --i;
    }
    if (i != m_PlaybackIndex[bufferName].end())
    {
      if (m_Buffers.find(bufferNameAsStdString) == m_Buffers.end())
      {
        std::unique_ptr<niftk::IGIDataSourceRingBuffer> newBuffer(
              new niftk::IGIDataSourceRingBuffer(m_Tracker->GetPreferredFramesPerSecond() * 2));
        newBuffer->SetLagInMilliseconds(m_Lag);
        m_Buffers.insert(std::make_pair(bufferNameAsStdString, std::move(newBuffer)));
      }

      if (m_Buffers.find(bufferNameAsStdString) == m_Buffers.end())
      {
        std::ostringstream  filename;
        filename << this->GetPlaybackDirectory().toStdString()
                 << niftk::GetPreferredSlash().toStdString()
                 << bufferName.toStdString()
                 << niftk::GetPreferredSlash().toStdString()
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

          niftk::IGITrackerDataType *trackerData = new niftk::IGITrackerDataType();
          trackerData->SetTimeStampInNanoSeconds(*i);
          trackerData->SetTrackingMatrix(matrix);
          trackerData->SetFrameId(m_FrameId++);
          trackerData->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
          trackerData->SetShouldBeSaved(false);

          std::unique_ptr<niftk::IGIDataType> wrapper(trackerData);

          // Buffer itself should be threadsafe, so I'm not locking anything here.
          m_Buffers[bufferNameAsStdString]->AddToBuffer(wrapper);

        } // end if file open
      } // end if item not already in buffer
    } // end: if we found a valid item to playback

    this->SetStatus("Playing back");

  } // end: for each buffer in playback index
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::GrabData()
{
  QMutexLocker locker(&m_Lock);

  if (this->GetIsPlayingBack())
  {
    return;
  }

  if (m_Tracker.IsNull())
  {
    mitkThrow() << "Tracker is null. This should not happen! It's a programming bug.";
  }

  niftk::IGIDataSourceI::IGITimeType timeCreated = this->GetTimeStampInNanoseconds();

  std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > result = m_Tracker->GetTrackingData();
  if (!result.empty())
  {
    std::map<std::string, vtkSmartPointer<vtkMatrix4x4> >::iterator iter;
    for (iter = result.begin(); iter != result.end(); ++iter)
    {
      std::string toolName = (*iter).first;

      niftk::IGITrackerDataType *trackerData = new niftk::IGITrackerDataType();
      trackerData->SetToolName(toolName);
      trackerData->SetTrackingMatrix((*iter).second);
      trackerData->SetTimeStampInNanoSeconds(timeCreated);
      trackerData->SetFrameId(m_FrameId++);
      trackerData->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
      trackerData->SetShouldBeSaved(this->GetIsRecording());

      std::unique_ptr<niftk::IGIDataType> wrapper(trackerData);

      if (m_Buffers.find(toolName) == m_Buffers.end())
      {
        std::unique_ptr<niftk::IGIDataSourceRingBuffer> newBuffer(
              new niftk::IGIDataSourceRingBuffer(m_Tracker->GetPreferredFramesPerSecond() * 2));
        newBuffer->SetLagInMilliseconds(m_Lag);
        m_Buffers.insert(std::make_pair(toolName, std::move(newBuffer)));
      }

      // Save synchronously.
      // This has the side effect that if saving is too slow,
      // the QTimers just won't keep up, and start missing pulses.
      if (this->GetIsRecording())
      {
        this->SaveItem(wrapper);
        this->SetStatus("Saving");
      }
      else
      {
        this->SetStatus("Grabbing");
      }
      m_Buffers[toolName]->AddToBuffer(wrapper);
    }
  }
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::SaveItem(const std::unique_ptr<niftk::IGIDataType>& item)
{

  niftk::IGITrackerDataType* data = dynamic_cast<niftk::IGITrackerDataType*>(item.get());
  if (data == nullptr)
  {
    mitkThrow() << "Failed to save IGITrackerDataType as the data received was the wrong type!";
  }

  vtkSmartPointer<vtkMatrix4x4> matrix = data->GetTrackingMatrix();
  if (matrix == NULL)
  {
    mitkThrow() << "Failed to save IGITrackerDataType as the tracking matrix was NULL!";
  }

  QString directoryPath = this->GetRecordingDirectory();

  QString toolPath = directoryPath
      + niftk::GetPreferredSlash()
      + QString::fromStdString(data->GetToolName())
      + niftk::GetPreferredSlash();

  QDir directory(toolPath);
  if (directory.mkpath(toolPath))
  {
    QString fileName =  toolPath + QDir::separator() + tr("%1.txt").arg(data->GetTimeStampInNanoSeconds());

    bool success = SaveVtkMatrix4x4ToFile(fileName.toStdString(), *matrix);
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
std::vector<IGIDataItemInfo> MITKTrackerDataSourceService::Update(const niftk::IGIDataSourceI::IGITimeType& time)
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
  if (m_Buffers.empty())
  {
    return infos;
  }

  if (!this->GetShouldUpdate())
  {
    return infos;
  }

  std::map<std::string, std::unique_ptr<niftk::IGIDataSourceRingBuffer> >::iterator iter;
  for (iter = m_Buffers.begin(); iter != m_Buffers.end(); ++iter)
  {
    std::string bufferName = iter->first;

    if (m_Buffers[bufferName]->GetBufferSize() == 0)
    {
      continue;
    }

    if(m_Buffers[bufferName]->GetFirstTimeStamp() > time)
    {
      continue;
    }

    bool gotFromBuffer = m_Buffers[bufferName]->CopyOutItem(time, m_CachedDataType);
    if (!gotFromBuffer)
    {
      MITK_INFO << "MITKTrackerDataSourceService: Failed to find data for time:" << time;
      return infos;
    }

    mitk::DataNode::Pointer node = this->GetDataNode(QString::fromStdString(bufferName));
    if (node.IsNull())
    {
      // The above call to GetDataNode should always retrieve one, or create it.
      mitkThrow() << "Can't find mitk::DataNode with name " << bufferName;
    }

    CoordinateAxesData::Pointer coords = dynamic_cast<CoordinateAxesData*>(node->GetData());
    if (coords.IsNull())
    {
      coords = CoordinateAxesData::New();

      // We remove and add to trigger the NodeAdded event,
      // which is not emmitted if the node was added with no data.
      this->GetDataStorage()->Remove(node);
      node->SetData(coords);
      this->GetDataStorage()->Add(node);
    }

    vtkSmartPointer<vtkMatrix4x4> matrix = m_CachedDataType.GetTrackingMatrix();
    coords->SetVtkMatrix(*matrix);

    // We tell the node that it is modified so the next rendering event
    // will redraw it. Triggering this does not in itself guarantee a re-rendering.
    coords->Modified();
    node->Modified();

    IGIDataItemInfo info;
    info.m_Name = this->GetName();
    info.m_FramesPerSecond = m_Buffers[bufferName]->GetFrameRate();
    info.m_IsLate = this->IsLate(time, m_CachedDataType.GetTimeStampInNanoSeconds());
    info.m_LagInMilliseconds = this->GetLagInMilliseconds(time, m_CachedDataType.GetTimeStampInNanoSeconds());
    infos.push_back(info);
  }

  return infos;
}

} // end namespace
