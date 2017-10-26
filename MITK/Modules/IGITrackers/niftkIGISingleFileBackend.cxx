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
#include <niftkMITKMathsUtils.h>
#include <niftkFileHelper.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace niftk
{

//-----------------------------------------------------------------------------
IGISingleFileBackend::IGISingleFileBackend(QString name, mitk::DataStorage::Pointer dataStorage)
: IGITrackerBackend(name, dataStorage)
, m_FileHeaderSize (256) //make the header largish so we can cope with later additions
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
    trackerData->SetDuration(duration);
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
  m_PlaybackDirectory = directoryName;
}


//-----------------------------------------------------------------------------
void IGISingleFileBackend::PlaybackData(const niftk::IGIDataSourceI::IGITimeType& duration,
                                        const niftk::IGIDataSourceI::IGITimeType& requestedTimeStamp)
{
  if (m_PlaybackIndex.empty())
  {
    mitkThrow() << "Empty m_PlaybackIndex, which must be a programming bug!";
  }

  // Remember we have multiple buffers!
  // This will find us the timestamp right before the requested one, in each buffer.
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
          mitk::Point4D rotation = (*i).second.first;
          mitk::Vector3D translation = (*i).second.second;

          niftk::IGITrackerDataType *trackerData = new niftk::IGITrackerDataType();
          trackerData->SetTimeStampInNanoSeconds((*i).first);
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
IGISingleFileBackend::PlaybackTransformType
IGISingleFileBackend::ParseFile(const QString& fileName)
{
  PlaybackTransformType result;
  std::ifstream ifs(fileName.toStdString(), std::ios::binary | std::ios::in);
  if (ifs.is_open())
  {
    try
    {
      this->CheckFileHeader(ifs);
    }
    catch ( std::exception& e )
    {
      mitkThrow() << fileName.toStdString() << "Does not appear to be a valid tracking data file." << e.what();
    }
    niftk::IGIDataSourceI::IGITimeType time;
    std::pair<mitk::Point4D, mitk::Vector3D> transform;
    while (ifs.good())
    {
      ifs.read(reinterpret_cast<char*>(&time), sizeof(niftk::IGIDataSourceI::IGITimeType));
      ifs.read(reinterpret_cast<char*>(&transform.first[0]), sizeof(transform.first[0]));
      ifs.read(reinterpret_cast<char*>(&transform.first[1]), sizeof(transform.first[1]));
      ifs.read(reinterpret_cast<char*>(&transform.first[2]), sizeof(transform.first[2]));
      ifs.read(reinterpret_cast<char*>(&transform.first[3]), sizeof(transform.first[3]));
      ifs.read(reinterpret_cast<char*>(&transform.second[0]), sizeof(transform.second[0]));
      ifs.read(reinterpret_cast<char*>(&transform.second[1]), sizeof(transform.second[1]));
      ifs.read(reinterpret_cast<char*>(&transform.second[2]), sizeof(transform.second[2]));
      if (ifs.good())
      {
        result.insert(std::move(std::make_pair(time, transform)));
      }
    }
  }
  return std::move(result);
}

//-----------------------------------------------------------------------------
void IGISingleFileBackend::CheckFileHeader ( std::ifstream& ifs )
{
  std::string header;
  ifs.read (reinterpret_cast<char*>(&header[0]),m_FileHeaderSize);
  if ( ! ifs.good () )
  {
    mitkThrow() << "Problem checking file header.";
  }
  std::string expectedHeader = this->GetFileHeader();
  if ( header.compare ( 0, expectedHeader.length(), expectedHeader ) != 0 )
  {
    mitkThrow() << "Not a valid tracking file";
  }
}

//-----------------------------------------------------------------------------
std::string IGISingleFileBackend::GetFileHeader ( )
{
  std::string header;
  std::stringstream toHeader;
  boost::property_tree::ptree pt;
  pt.add ("NifTK_TRQD.version", 0.0);
  boost::property_tree::xml_writer_settings<std::string> settings(' ',2);
  boost::property_tree::write_xml (toHeader, pt, settings);
  header = toHeader.str();
  return header;
}

//-----------------------------------------------------------------------------
bool IGISingleFileBackend::ProbeRecordedData(const QString& directoryName,
                                             niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                             niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  niftk::IGIDataSourceI::IGITimeType  firstTimeStampFound
    = std::numeric_limits<niftk::IGIDataSourceI::IGITimeType>::max();

  niftk::IGIDataSourceI::IGITimeType  lastTimeStampFound
    = std::numeric_limits<niftk::IGIDataSourceI::IGITimeType>::min();

  // Get all transform files.
  std::vector<std::string> files = niftk::FindFilesWithGivenExtension(directoryName.toStdString(), ".tqrt");
  if (files.empty())
  {
    return false;
  }
  for (int i = 0; i < files.size(); i++)
  {
    std::string fileName = files[i];
    PlaybackTransformType map = this->ParseFile(QString::fromStdString(fileName));
    if (!map.empty())
    {
      niftk::IGIDataSourceI::IGITimeType firstTimeStamp = (*map.begin()).first;
      if (firstTimeStamp < firstTimeStampFound)
      {
        firstTimeStampFound = firstTimeStamp;
      }
      niftk::IGIDataSourceI::IGITimeType lastTimeStamp;
      if (map.size() > 1)
      {
        lastTimeStamp = (*map.rbegin()).first;
      }
      else
      {
        lastTimeStamp = (*map.begin()).first;
      }
      if (lastTimeStamp > lastTimeStampFound)
      {
        lastTimeStampFound = lastTimeStamp;
      }

      MITK_INFO << "IGISingleFileBackend: Parsed:" << fileName
                << ", min=" << firstTimeStamp
                << ", max=" << lastTimeStamp;
    }
    MITK_INFO << "IGISingleFileBackend: Probed:" << directoryName.toStdString()
              << ", min=" << firstTimeStampFound
              << ", max=" << lastTimeStampFound;
  }

  if (firstTimeStampInStore)
  {
    *firstTimeStampInStore = firstTimeStampFound;
  }
  if (lastTimeStampInStore)
  {
    *lastTimeStampInStore = lastTimeStampFound;
  }

  return firstTimeStampFound != std::numeric_limits<niftk::IGIDataSourceI::IGITimeType>::max();
}


//-----------------------------------------------------------------------------
void IGISingleFileBackend::StopRecording()
{
  m_OpenFiles.clear();
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

  std::string toolName = data->GetToolName();

  QString toolPath = directoryName
      + niftk::GetPreferredSlash()
      + QString::fromStdString(toolName)
      + niftk::GetPreferredSlash();

  QDir directory(toolPath);
  if (!directory.exists())
  {
    if (!directory.mkpath(toolPath))
    {
      mitkThrow() << "Failed to save IGITrackerDataType as could not create " << toolPath.toStdString();
    }
  }

  QString fileName =  toolPath + QDir::separator() + QString::fromStdString(toolName) + QString(".tqrt");
  std::string fileNameAsString = fileName.toStdString();

  // Open file if its not in our map.
  if (m_OpenFiles.find(toolName) == m_OpenFiles.end())
  {
    std::unique_ptr<ofstream> ofs(new ofstream());
    ofs->open(fileNameAsString, std::ios::binary | std::ios::out);
    if (!ofs->is_open())
    {
      mitkThrow() << "Failed to open file:" << fileNameAsString << " for saving data.";
    }
    std::string header = this->GetFileHeader();
    MITK_INFO << "Writing header " << header;
    //ofs->write(reinterpret_cast<char*>(&header), m_FileHeaderSize);
    *ofs << header;
    m_OpenFiles.insert(std::move(std::make_pair(toolName, std::move(ofs))));
  }

  // Write data to file.
  mitk::Point4D rotation;
  mitk::Vector3D translation;
  data->GetTransform(rotation, translation);

  niftk::IGIDataSourceI::IGITimeType time = data->GetTimeStampInNanoSeconds();

  (*m_OpenFiles[toolName]).write(reinterpret_cast<char*>(&time), sizeof(niftk::IGIDataSourceI::IGITimeType));
  (*m_OpenFiles[toolName]).write(reinterpret_cast<char*>(&rotation[0]), sizeof(rotation[0]));
  (*m_OpenFiles[toolName]).write(reinterpret_cast<char*>(&rotation[1]), sizeof(rotation[1]));
  (*m_OpenFiles[toolName]).write(reinterpret_cast<char*>(&rotation[2]), sizeof(rotation[2]));
  (*m_OpenFiles[toolName]).write(reinterpret_cast<char*>(&rotation[3]), sizeof(rotation[3]));
  (*m_OpenFiles[toolName]).write(reinterpret_cast<char*>(&translation[0]), sizeof(translation[0]));
  (*m_OpenFiles[toolName]).write(reinterpret_cast<char*>(&translation[1]), sizeof(translation[1]));
  (*m_OpenFiles[toolName]).write(reinterpret_cast<char*>(&translation[2]), sizeof(translation[2]));

  data->SetIsSaved(true);
}


//-----------------------------------------------------------------------------
IGISingleFileBackend::PlaybackIndexType
IGISingleFileBackend::GetPlaybackIndex(const QString& directoryName)
{
  PlaybackIndexType playbackIndex;

  std::vector<std::string> files = niftk::FindFilesWithGivenExtension(directoryName.toStdString(), ".tqrt");
  if (files.empty())
  {
    return std::move(playbackIndex);
  }
  for (int i = 0; i < files.size(); i++)
  {
    std::string fileName = files[i];
    std::string base = niftk::Basename(fileName);

    PlaybackTransformType map = this->ParseFile(QString::fromStdString(fileName));
    if (map.size() > 0)
    {
      MITK_INFO << "IGISingleFileBackend: Loaded " << fileName << ", with " << map.size() << " transforms";
      playbackIndex.insert(std::move(std::make_pair(base,
                                                    std::move(map)
                                                   )
                                     )
                           );
    }
  }

  MITK_INFO << "IGISingleFileBackend: Stored " << playbackIndex.size() << " buffers.";

  return std::move(playbackIndex);
}


} // end namespace
