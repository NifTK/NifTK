/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "AudioDataSource.h"
#include <stdexcept>
#include <cassert>
#include <QAudioInput>
#include <QAudioDeviceInfo>


//-----------------------------------------------------------------------------
AudioDataSource::AudioDataSource(mitk::DataStorage* storage)
  : QmitkIGILocalDataSource(storage)
  , m_InputDevice(0)
  , m_WasSavingMessagesPreviously(false)
{
  SetName("AudioDataSource");
  SetStatus("Initialising...");

  // these should be updated based on QAudioDeviceInfo::defaultInputDevice()
  SetType("Microphone");
  SetDescription("NVidia SDI");
}


//-----------------------------------------------------------------------------
AudioDataSource::~AudioDataSource()
{
}


//-----------------------------------------------------------------------------
void AudioDataSource::SetAudioDevice(QAudioDeviceInfo* device)
{
}


//-----------------------------------------------------------------------------
bool AudioDataSource::GetSaveInBackground() const
{
  return false;
}


//-----------------------------------------------------------------------------
bool AudioDataSource::CanHandleData(mitk::IGIDataType* data) const
{
  return true;
}


//-----------------------------------------------------------------------------
bool AudioDataSource::ProbeRecordedData(const std::string& path, igtlUint64* firstTimeStampInStore, igtlUint64* lastTimeStampInStore)
{
  return false;
}


//-----------------------------------------------------------------------------
void AudioDataSource::StartPlayback(const std::string& path, igtlUint64 firstTimeStamp, igtlUint64 lastTimeStamp)
{
  throw std::logic_error("Not supported");
}


//-----------------------------------------------------------------------------
void AudioDataSource::StopPlayback()
{
  // playback not supported (yet), so cannot stop it.
  assert(false);
}


//-----------------------------------------------------------------------------
void AudioDataSource::PlaybackData(igtlUint64 requestedTimeStamp)
{
  throw std::logic_error("Not supported");
}


//-----------------------------------------------------------------------------
void AudioDataSource::GrabData()
{
  // FIXME
}


//-----------------------------------------------------------------------------
bool AudioDataSource::SaveData(mitk::IGIDataType* data, std::string& outputFileName)
{
  // cannot record while playing back
  assert(!GetIsPlayingBack());

  // are we starting to record now
  if (m_WasSavingMessagesPreviously == false)
  {
    // FIXME: use qt for this
    //        see https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/2546
    SYSTEMTIME  now;
    // we used to have utc here but all the other data sources use local time too.
    GetLocalTime(&now);

    std::string directoryPath = this->GetSaveDirectoryName();
    QDir directory(QString::fromStdString(directoryPath));
    if (directory.mkpath(QString::fromStdString(directoryPath)))
    {
      std::ostringstream    filename;
      filename << directoryPath << "/capture-" 
        << now.wYear << '_' << now.wMonth << '_' << now.wDay << '-' << now.wHour << '_' << now.wMinute << '_' << now.wSecond;

      std::string filenamebase = filename.str();

      // FIXME: do something, write some index if necessary for seeking.

      // FIXME: init m_InputDevice
    }
  }


  // FIXME:

  return false;
}


//-----------------------------------------------------------------------------
bool AudioDataSource::Update(mitk::IGIDataType* data)
{
  return true;
}
