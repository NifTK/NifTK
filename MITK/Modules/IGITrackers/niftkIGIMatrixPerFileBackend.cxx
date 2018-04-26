/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIMatrixPerFileBackend.h"
#include <niftkIGIDataSourceUtils.h>
#include <niftkFileIOUtils.h>
#include <niftkMITKMathsUtils.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIMatrixPerFileBackend::IGIMatrixPerFileBackend(QString name, mitk::DataStorage::Pointer dataStorage)
: IGITrackerBackend(name, dataStorage)
{
  m_CachedTransformForSaving = vtkSmartPointer<vtkMatrix4x4>::New();
  m_CachedTransformForSaving->Identity();
}


//-----------------------------------------------------------------------------
IGIMatrixPerFileBackend::~IGIMatrixPerFileBackend()
{
}


//-----------------------------------------------------------------------------
void IGIMatrixPerFileBackend::AddData(const QString& directoryName,
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
void IGIMatrixPerFileBackend::StartPlayback(const QString& directoryName,
                                            const niftk::IGIDataSourceI::IGITimeType& firstTimeStamp,
                                            const niftk::IGIDataSourceI::IGITimeType& lastTimeStamp)
{
  m_Buffers.clear();
  m_PlaybackIndex = this->GetPlaybackIndex(directoryName);
  m_PlaybackDirectory = directoryName;
}


//-----------------------------------------------------------------------------
void IGIMatrixPerFileBackend::PlaybackData(const niftk::IGIDataSourceI::IGITimeType& duration,
                                           const niftk::IGIDataSourceI::IGITimeType& requestedTimeStamp)
{
  if (m_PlaybackIndex.empty())
  {
    mitkThrow() << "Empty m_PlaybackIndex, which must be a programming bug!";
  }

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
              new niftk::IGIDataSourceRingBuffer(this->GetExpectedFramesPerSecond() * 2));
        newBuffer->SetLagInMilliseconds(m_Lag);
        m_Buffers.insert(std::make_pair(bufferNameAsStdString, std::move(newBuffer)));
      }

      if (m_Buffers.find(bufferNameAsStdString) != m_Buffers.end())
      {
        std::ostringstream  filename;
        filename << m_PlaybackDirectory.toStdString()
                 << niftk::GetPreferredSlash().toStdString()
                 << bufferName.toStdString()
                 << niftk::GetPreferredSlash().toStdString()
                 << (*i)
                 << ".txt";

        std::ifstream file(filename.str().c_str());
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

          mitk::Point4D rotation;
          mitk::Vector3D translation;
          niftk::ConvertMatrixToRotationAndTranslation(*matrix, rotation, translation);

          niftk::IGITrackerDataType *trackerData = new niftk::IGITrackerDataType();
          trackerData->SetTimeStampInNanoSeconds(*i);
          trackerData->SetTransform(rotation, translation);
          trackerData->SetFrameId(m_FrameId++);
          trackerData->SetDuration(duration);
          trackerData->SetShouldBeSaved(false);

          std::unique_ptr<niftk::IGIDataType> wrapper(trackerData);

          // Buffer itself should be threadsafe, so I'm not locking anything here.
          m_Buffers[bufferNameAsStdString]->AddToBuffer(wrapper);

        } // end: if file open
      } // end: if item not already in buffer
    } // end: if we found a valid item to playback
  } // end: for each buffer in playback index
}


//-----------------------------------------------------------------------------
void IGIMatrixPerFileBackend::StopPlayback()
{
  m_PlaybackIndex.clear();
  m_Buffers.clear();
}


//-----------------------------------------------------------------------------
bool IGIMatrixPerFileBackend::ProbeRecordedData(const QString& directoryName,
                                                niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                                niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  return niftk::ProbeRecordedData(directoryName, QString(".txt"), firstTimeStampInStore, lastTimeStampInStore);
}


//-----------------------------------------------------------------------------
QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> >
IGIMatrixPerFileBackend::GetPlaybackIndex(const QString& directoryName)
{
  QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> > bufferToTimeStamp;
  QMap<QString, QHash<niftk::IGIDataSourceI::IGITimeType, QStringList> > bufferToTimeStampToFileNames;

  niftk::GetPlaybackIndex(directoryName, QString(".txt"), bufferToTimeStamp, bufferToTimeStampToFileNames);
  return bufferToTimeStamp;
}


//-----------------------------------------------------------------------------
void IGIMatrixPerFileBackend::SaveItem(const QString& directoryName,
                                       const std::unique_ptr<niftk::IGIDataType>& item)
{
  niftk::IGITrackerDataType* data = dynamic_cast<niftk::IGITrackerDataType*>(item.get());
  if (data == nullptr)
  {
    mitkThrow() << "Failed to save IGITrackerDataType as the data received was the wrong type!";
  }

  QString toolPath = directoryName
      + niftk::GetPreferredSlash()
      + QString::fromStdString(data->GetToolName())
      + niftk::GetPreferredSlash();

  QDir directory(toolPath);
  if (!directory.exists())
  {
    if (!directory.mkpath(toolPath))
    {
      mitkThrow() << "Failed to save IGITrackerDataType as could not create " << toolPath.toStdString();
    }
  }

  QString fileName =  toolPath + QDir::separator() + QObject::tr("%1.txt").arg(data->GetTimeStampInNanoSeconds());

  mitk::Point4D rotation;
  mitk::Vector3D translation;
  data->GetTransform(rotation, translation);
  niftk::ConvertRotationAndTranslationToMatrix(rotation, translation, *m_CachedTransformForSaving);
  bool success = SaveVtkMatrix4x4ToFile(fileName.toStdString(), *m_CachedTransformForSaving);

  if (!success)
  {
    mitkThrow() << "Failed to save IGITrackerDataType to " << fileName.toStdString();
  }
  data->SetIsSaved(true);
}

} // end namespace
