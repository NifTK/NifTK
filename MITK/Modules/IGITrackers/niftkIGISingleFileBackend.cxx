/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGISingleFileBackend.h"
#include <niftkIGIDataSourceUtils.h>
#include <niftkFileIOUtils.h>
#include <niftkMITKMathsUtils.h>

#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
IGISingleFileBackend::IGISingleFileBackend(QString name, mitk::DataStorage::Pointer dataStorage)
: IGITrackerBackend(name, dataStorage)
{
}


//-----------------------------------------------------------------------------
IGISingleFileBackend::~IGISingleFileBackend()
{
}


//-----------------------------------------------------------------------------
void IGISingleFileBackend::AddData(const QString& directoryName,
                                      const bool& isRecording,
                                      const niftk::IGIDataSourceI::IGITimeType& duration,
                                      const niftk::IGIDataSourceI::IGITimeType& timeStamp,
                                      const std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >& data)
{
  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >::const_iterator iter;
  for (iter = data.begin(); iter != data.end(); ++iter)
  {
    std::string toolName = (*iter).first;

    niftk::IGITrackerDataType *trackerData = new niftk::IGITrackerDataType();
    trackerData->SetToolName(toolName);
    trackerData->SetTransform((*iter).second.first, (*iter).second.second);
    trackerData->SetTimeStampInNanoSeconds(timeStamp);
    trackerData->SetFrameId(m_FrameId++);
    trackerData->SetDuration(duration); // nanoseconds
    trackerData->SetShouldBeSaved(isRecording);

    std::unique_ptr<niftk::IGIDataType> wrapper(trackerData);

    if (m_Buffers.find(toolName) == m_Buffers.end())
    {
      std::unique_ptr<niftk::IGIDataSourceRingBuffer> newBuffer(
            new niftk::IGIDataSourceRingBuffer(this->GetExpectedFramesPerSecond() * 2));
      newBuffer->SetLagInMilliseconds(m_Lag);
      m_Buffers.insert(std::make_pair(toolName, std::move(newBuffer)));
    }

    if (isRecording)
    {
      this->SaveItem(directoryName, wrapper);
    }

    m_Buffers[toolName]->AddToBuffer(wrapper);
  }
}


//-----------------------------------------------------------------------------
void IGISingleFileBackend::StartPlayback(const QString& directoryName,
                                         const niftk::IGIDataSourceI::IGITimeType& firstTimeStamp,
                                         const niftk::IGIDataSourceI::IGITimeType& lastTimeStamp)
{
  m_Buffers.clear();
  m_PlaybackIndex = this->GetPlaybackIndex(directoryName);
}


//-----------------------------------------------------------------------------
void IGISingleFileBackend::PlaybackData(const QString& directoryName,
                                        const niftk::IGIDataSourceI::IGITimeType& duration,
                                        const niftk::IGIDataSourceI::IGITimeType& requestedTimeStamp)
{
  assert(m_PlaybackIndex.size() > 0); // Should have failed probing if no data.

  // This will find us the timestamp right before the requested one.
  // Remember we have multiple buffers!
  PlaybackIndexType::iterator playbackIter;
  for(playbackIter = m_PlaybackIndex.begin(); playbackIter != m_PlaybackIndex.end(); ++playbackIter)
  {
    std::string bufferName = (*playbackIter).first;

    PlaybackTransformType::const_iterator i =
      m_PlaybackIndex[bufferName].upper_bound(requestedTimeStamp);

    if (i != m_PlaybackIndex[bufferName].begin())
    {
      --i;
    }
    if (i != m_PlaybackIndex[bufferName].end())
    {
      if (m_Buffers.find(bufferName) == m_Buffers.end())
      {
        std::unique_ptr<niftk::IGIDataSourceRingBuffer> newBuffer(
              new niftk::IGIDataSourceRingBuffer(this->GetExpectedFramesPerSecond() * 2));
        newBuffer->SetLagInMilliseconds(m_Lag);
        m_Buffers.insert(std::make_pair(bufferName, std::move(newBuffer)));
      }

      if (m_Buffers.find(bufferName) != m_Buffers.end())
      {
          mitk::Point4D rotation;
          mitk::Vector3D translation;

          niftk::IGITrackerDataType *trackerData = new niftk::IGITrackerDataType();
          //trackerData->SetTimeStampInNanoSeconds(*i);
          trackerData->SetTransform(rotation, translation);
          trackerData->SetFrameId(m_FrameId++);
          trackerData->SetDuration(duration);
          trackerData->SetShouldBeSaved(false);

          std::unique_ptr<niftk::IGIDataType> wrapper(trackerData);

          // Buffer itself should be threadsafe, so I'm not locking anything here.
          m_Buffers[bufferName]->AddToBuffer(wrapper);

      } // end: if item not already in buffer
    } // end: if we found a valid item to playback
  } // end: for each buffer in playback index
}


//-----------------------------------------------------------------------------
void IGISingleFileBackend::StopPlayback()
{
  m_PlaybackIndex.clear();
  m_Buffers.clear();
}


//-----------------------------------------------------------------------------
bool IGISingleFileBackend::ProbeRecordedData(const QString& directoryName,
                                             niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                             niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  return false;
}


//-----------------------------------------------------------------------------
void IGISingleFileBackend::SaveItem(const QString& directoryName,
                                    const std::unique_ptr<niftk::IGIDataType>& item)
{
  niftk::IGITrackerDataType* data = dynamic_cast<niftk::IGITrackerDataType*>(item.get());
  if (data == nullptr)
  {
    mitkThrow() << "Failed to save IGITrackerDataType as the data received was the wrong type!";
  }

  data->SetIsSaved(true);
}


//-----------------------------------------------------------------------------
IGISingleFileBackend::PlaybackIndexType
IGISingleFileBackend::GetPlaybackIndex(const QString& directoryName)
{
  PlaybackIndexType playbackIndex;
  return playbackIndex;
}


} // end namespace
