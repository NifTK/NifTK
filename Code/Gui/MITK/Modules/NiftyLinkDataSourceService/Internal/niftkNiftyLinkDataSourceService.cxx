/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyLinkDataSourceService.h"
#include "niftkNiftyLinkDataType.h"

#include <mitkCoordinateAxesData.h>

#include <igtlTrackingDataMessage.h>
#include <igtlImageMessage.h>
#include <igtlStringMessage.h>

#include <vtkSmartPointer.h>

#include <QDir>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker NiftyLinkDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
NiftyLinkDataSourceService::NiftyLinkDataSourceService(
    QString name,
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage
    )
: IGIDataSource((name + QString("-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_FrameId(0)
, m_BackgroundDeleteThread(NULL)
, m_Lag(0)
{
  qRegisterMetaType<niftk::NiftyLinkMessageContainer::Pointer>("niftk::NiftyLinkMessageContainer::Pointer");

  QString deviceName = this->GetName();
  m_SourceNumber = (deviceName.remove(0, name.length() + 1)).toInt();

  m_MessageReceivedTimeStamp = igtl::TimeStamp::New();
  m_MessageReceivedTimeStamp->GetTime();

  this->SetStatus("Initialising");

  // In contrast with other sources, like a frame grabber, where you
  // know the expected frame rate, a network source could be anything.
  // Lets assume for now:
  //   Vicra = 20 fps, Spectra, Aurora = faster.
  //   Ultrasonix = 20 fpas, or faster.
  // So, 20 fps = 50 ms.
  this->SetTimeStampTolerance(50*1000000);
  this->SetProperties(properties);
  this->SetShouldUpdate(true);
  this->StartCapturing();

  m_BackgroundDeleteThread = new niftk::IGIDataSourceBackgroundDeleteThread(NULL, this);
  m_BackgroundDeleteThread->SetInterval(1000); // try deleting data every 1 second.
  m_BackgroundDeleteThread->start();
  if (!m_BackgroundDeleteThread->isRunning())
  {
    mitkThrow() << "Failed to start background deleting thread";
  }

  m_BackgroundSaveThread = new niftk::IGIDataSourceBackgroundSaveThread(NULL, this);
  m_BackgroundSaveThread->SetInterval(500); // try deleting data every 0.5 second.
  m_BackgroundSaveThread->start();
  if (!m_BackgroundSaveThread->isRunning())
  {
    mitkThrow() << "Failed to start background save thread";
  }

  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
NiftyLinkDataSourceService::~NiftyLinkDataSourceService()
{
  this->StopCapturing();

  s_Lock.RemoveSource(m_SourceNumber);

  m_BackgroundDeleteThread->ForciblyStop();
  delete m_BackgroundDeleteThread;

  m_BackgroundSaveThread->ForciblyStop();
  delete m_BackgroundSaveThread;
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  // In contrast say, to the OpenCV source, we don't set the lag
  // directly on the buffer because, there may be no buffers present
  // at the time this method is called. For example, you could
  // have created a tracker, and no tracked objects are placed within
  // the field of view, thereby no tracking matrices would have been generated.
  if (properties.contains("lag"))
  {
    int milliseconds = (properties.value("lag")).toInt();
    m_Lag = milliseconds;

    MITK_INFO << "NiftyLinkDataSourceService(" << this->GetName().toStdString()
              << "): set lag to " << milliseconds << " ms.";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties NiftyLinkDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  props.insert("lag", m_Lag);

  MITK_INFO << "NiftyLinkDataSourceService:(" << this->GetName().toStdString()
            << "): Retrieved current value of lag as " << m_Lag << " ms.";

  return props;
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::StartCapturing()
{
  this->SetStatus("Capturing");
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::StopCapturing()
{
  this->SetStatus("Stopped");
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::CleanBuffer()
{
  // Buffers should be threadsafe. Clean all buffers.
  QMap<QString, niftk::IGIWaitForSavedDataSourceBuffer::Pointer>::iterator iter;
  for (iter = m_Buffers.begin(); iter != m_Buffers.end(); iter++)
  {
    (*iter)->CleanBuffer();
  }
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::SaveBuffer()
{
  // Buffers should be threadsafe. Save all buffers.
  QMap<QString, niftk::IGIWaitForSavedDataSourceBuffer::Pointer>::iterator iter;
  for (iter = m_Buffers.begin(); iter != m_Buffers.end(); iter++)
  {
    (*iter)->SaveBuffer();
  }
}


//-----------------------------------------------------------------------------
QString NiftyLinkDataSourceService::GetRecordingDirectoryName()
{
  return this->GetRecordingLocation()
      + this->GetPreferredSlash()
      + this->GetName()
      ;
}


//-----------------------------------------------------------------------------
QMap<QString, std::set<niftk::IGIDataType::IGITimeType> >  NiftyLinkDataSourceService::GetPlaybackIndex(QString directory)
{
  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> > result;
/*
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
*/
  return result;
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataType::IGITimeType lastTimeStamp)
{
  QMutexLocker locker(&m_Lock);

  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);

  m_Buffers.clear();
  m_PlaybackIndex = this->GetPlaybackIndex(this->GetRecordingDirectoryName());
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::StopPlayback()
{
  QMutexLocker locker(&m_Lock);

  m_PlaybackIndex.clear();
  m_Buffers.clear();

  IGIDataSource::StopPlayback();
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp)
{
  assert(this->GetIsPlayingBack());
  assert(m_PlaybackIndex.size() > 0); // Should have failed probing if no data.
/*
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
*/
  this->SetStatus("Playing back");
}


//-----------------------------------------------------------------------------
bool NiftyLinkDataSourceService::ProbeRecordedData(const QString& path,
                                                     niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                                                     niftk::IGIDataType::IGITimeType* lastTimeStampInStore)
{
  niftk::IGIDataType::IGITimeType  firstTimeStampFound = std::numeric_limits<niftk::IGIDataType::IGITimeType>::max();
  niftk::IGIDataType::IGITimeType  lastTimeStampFound  = std::numeric_limits<niftk::IGIDataType::IGITimeType>::min();
/*
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
*/
  return firstTimeStampFound != std::numeric_limits<niftk::IGIDataType::IGITimeType>::max();
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::SaveItem(niftk::IGIDataType::Pointer data)
{
  niftk::NiftyLinkDataType::Pointer niftyLinkType = dynamic_cast<niftk::NiftyLinkDataType*>(data.GetPointer());
  if (niftyLinkType.IsNull())
  {
    mitkThrow() << this->GetName().toStdString() << ":Received null data?!?";
  }

  if (niftyLinkType->GetIsSaved())
  {
    return;
  }

/*
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
*/
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> NiftyLinkDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;

  // This loads playback-data into the buffers, so must
  // come before the check for empty buffer.
  if (this->GetIsPlayingBack())
  {
    this->PlaybackData(time);
  }

  // Early exit if no buffers, which means that
  // the source is created, but has not recorded any data yet.

  if (m_Buffers.isEmpty())
  {
    IGIDataItemInfo info;
    info.m_Name = this->GetName();
    info.m_Status = this->GetStatus();
    info.m_ShouldUpdate = this->GetShouldUpdate();
    info.m_Description = "Network Source";
    info.m_FramesPerSecond = 0;
    info.m_IsLate = false;
    info.m_LagInMilliseconds = 0;
    infos.push_back(info);
    return infos;
  }

  // i.e. we are frozen. No update.
  if (!this->GetShouldUpdate())
  {
    return infos;
  }

  QMap<QString, niftk::IGIWaitForSavedDataSourceBuffer::Pointer>::iterator iter;
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

    niftk::NiftyLinkDataType::Pointer dataType = dynamic_cast<niftk::NiftyLinkDataType*>(m_Buffers[bufferName]->GetItem(time).GetPointer());
    if (dataType.IsNull())
    {
      MITK_DEBUG << "Failed to find data for time " << time << ", size=" << m_Buffers[bufferName]->GetBufferSize() << ", last=" << m_Buffers[bufferName]->GetLastTimeStamp() << std::endl;
      continue;
    }

    niftk::NiftyLinkMessageContainer::Pointer msgContainer = dataType->GetMessageContainer();
    if (msgContainer.data() == NULL)
    {
      mitkThrow() << this->GetName().toStdString() << ":NiftyLinkDataType does not contain a NiftyLinkMessageContainer";
    }

    igtl::MessageBase::Pointer igtlMessage = msgContainer->GetMessage();
    if (igtlMessage.IsNull())
    {
      mitkThrow() << this->GetName().toStdString() << ":NiftyLinkMessageContainer contains a NULL igtl message";
    }

    igtl::StringMessage::Pointer stringMessage = dynamic_cast<igtl::StringMessage*>(igtlMessage.GetPointer());
    if (stringMessage.IsNotNull())
    {
      MITK_INFO << this->GetName().toStdString() << ":Received " << stringMessage->GetString();
      return infos;
    }

    igtl::TrackingDataMessage::Pointer trackingMessage = dynamic_cast<igtl::TrackingDataMessage*>(igtlMessage.GetPointer());
    if (trackingMessage.IsNotNull())
    {
      igtl::TrackingDataElement::Pointer tdata = igtl::TrackingDataElement::New();
      igtl::Matrix4x4 mat;
      QString toolName;
      vtkSmartPointer<vtkMatrix4x4> vtkMat = vtkSmartPointer<vtkMatrix4x4>::New();

      for (int i = 0; i < trackingMessage->GetNumberOfTrackingDataElements(); i++)
      {
        trackingMessage->GetTrackingDataElement(i, tdata);
        tdata->GetMatrix(mat);
        toolName = QString::fromStdString(tdata->GetName());

        mitk::DataNode::Pointer node = this->GetDataNode(toolName);
        if (node.IsNull())
        {
          mitkThrow() << this->GetName().toStdString() << ":Can't find mitk::DataNode with name " << toolName.toStdString();
        }

        mitk::CoordinateAxesData::Pointer coord = dynamic_cast<mitk::CoordinateAxesData*>(node->GetData());
        if (coord.IsNull())
        {
          coord = mitk::CoordinateAxesData::New();
          node->SetData(coord);
        }

        for (int r = 0; r < 4; r++)
        {
          for (int c = 0; c < 4; c++)
          {
            vtkMat->SetElement(r, c, mat[r][c]);
          }
        }
        coord->SetVtkMatrix(*vtkMat);
      }

      IGIDataItemInfo info;
      info.m_Name = toolName;
      info.m_Status = this->GetStatus();
      info.m_ShouldUpdate = this->GetShouldUpdate();
      info.m_FramesPerSecond = m_Buffers[bufferName]->GetFrameRate();
      info.m_Description = "Network Source";
      info.m_IsLate = this->IsLate(time, dataType->GetTimeStampInNanoSeconds());
      info.m_LagInMilliseconds = this->GetLagInMilliseconds(time, dataType->GetTimeStampInNanoSeconds());
      infos.push_back(info);
    } // end if its a tracking message

    igtl::ImageMessage::Pointer imgMsg = dynamic_cast<igtl::ImageMessage*>(igtlMessage.GetPointer());
    if (imgMsg.IsNotNull())
    {

    }

  }
  return infos;
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::MessageReceived(niftk::NiftyLinkMessageContainer::Pointer message)
{
  if (message.data() == NULL)
  {
    mitkThrow() << "Null message received, surely a programming bug?!?";
  }

  bool isRecording = false;
  {
    QMutexLocker locker(&m_Lock);
    if (this->GetIsPlayingBack())
    {
      return;
    }
    isRecording = this->GetIsRecording();
  }

  // Remember: network clients may send junk.
  // So don't throw an error, just log it and ignore.
  igtl::MessageBase::Pointer igtlMessage = message->GetMessage();
  if (igtlMessage.IsNull())
  {
    MITK_WARN << this->GetName().toStdString() << ":NiftyLinkMessageContainer contains a NULL igtl message.";
    return;
  }

  // Try to get the best time stamp available.
  // Remember: network clients could be rather
  // unreliable, or have incorrectly synched clock.
  niftk::IGIDataType::IGITimeType localTime = this->GetTimeStampInNanoseconds();
  message->GetTimeCreated(m_MessageReceivedTimeStamp);
  niftk::IGIDataType::IGITimeType timeCreated = m_MessageReceivedTimeStamp->GetTimeStampInNanoseconds();
  niftk::IGIDataType::IGITimeType timeToUse = 0;
  if (timeCreated > localTime  // if remote end is ahead, clock must be wrong.
      || timeCreated == 0      // if not specified, time data is useless.
      || timeCreated == std::numeric_limits<niftk::IGIDataType::IGITimeType>::min()
      || timeCreated == std::numeric_limits<niftk::IGIDataType::IGITimeType>::max()
      )
  {
    timeToUse = localTime;
  }
  else
  {
    timeToUse = timeCreated;
  }

  QString originator(igtlMessage->GetDeviceName());

  niftk::NiftyLinkDataType::Pointer wrapper = niftk::NiftyLinkDataType::New();
  wrapper->SetMessageContainer(message);
  wrapper->SetTimeStampInNanoSeconds(timeToUse);
  wrapper->SetFrameId(m_FrameId++);
  wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
  wrapper->SetShouldBeSaved(isRecording);

  if (!m_Buffers.contains(originator))
  {
    // So buffer requires a back-ground delete thread.
    niftk::IGIWaitForSavedDataSourceBuffer::Pointer newBuffer
        = niftk::IGIWaitForSavedDataSourceBuffer::New(100, this);

    m_Buffers.insert(originator, newBuffer);
  }

  if (isRecording)
  {
    // Save synchronously, within this thread (triggered from Network).
    if (wrapper->IsFastToSave())
    {
      this->SaveItem(wrapper.GetPointer());
      wrapper->SetIsSaved(true); // clear down happens in another thread.
    }
    else
    {
      // Save asynchronously in a background thread.
      wrapper->SetIsSaved(false);
    }
  }

  // I'm adding this last, so that the isSaved field is correct at the point
  // the item enters the buffer. This means the background delete thread and background
  // save thread won't know about it until it enters the buffer here.
  m_Buffers[originator]->AddToBuffer(wrapper.GetPointer());

  this->SetStatus("Grabbing");
}

} // end namespace
