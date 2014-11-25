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
AudioDataType::AudioDataType()
  : m_AudioBlob(0)
  , m_Length(0)
{
}


//-----------------------------------------------------------------------------
AudioDataType::~AudioDataType()
{
  delete m_AudioBlob;
}


//-----------------------------------------------------------------------------
void AudioDataType::SetBlob(const char* blob, std::size_t length)
{
  delete m_AudioBlob;
  m_AudioBlob = blob;
  m_Length = length;
}


//-----------------------------------------------------------------------------
AudioDataSource::AudioDataSource(mitk::DataStorage* storage)
  : QmitkIGILocalDataSource(storage)
  , m_InputDevice(0)
  , m_WasSavingMessagesPreviously(false)
{
  SetStatus("Initialising...");

  QList<QAudioDeviceInfo>   allDevices;// = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
//  foreach(QAudioDeviceInfo d, allDevices)
//  {
//    std::cerr << d.deviceName().toStdString() << std::endl;
//  }

  QAudioDeviceInfo  defaultDevice = allDevices.empty() ? QAudioDeviceInfo::defaultInputDevice() : allDevices.front();
  QAudioFormat      defaultFormat = defaultDevice.preferredFormat();

  SetAudioDevice(&defaultDevice, &defaultFormat);
}


//-----------------------------------------------------------------------------
AudioDataSource::~AudioDataSource()
{
  delete m_InputDevice;
  // we do not own m_InputStream!
}


//-----------------------------------------------------------------------------
void AudioDataSource::SetAudioDevice(QAudioDeviceInfo* device, QAudioFormat* format)
{
  assert(device != 0);
  assert(format != 0);

  // FIXME: disconnect previous audio device!


  try
  {

    m_InputDevice = new QAudioInput(*device, *format);
    bool ok = false;
    ok = QObject::connect(m_InputDevice, SIGNAL(stateChanged(QAudio::State)), this, SLOT(OnStateChanged(QAudio::State)));
    assert(ok);

    m_InputStream = m_InputDevice->start();
    ok = QObject::connect(m_InputStream, SIGNAL(readyRead()), this, SLOT(OnReadyRead()));
    assert(ok);

    SetType("QAudioInput");
    SetName(device->deviceName().toStdString());

    std::ostringstream    description;
    description << m_InputDevice->format().channels() << " channels @ " << m_InputDevice->format().sampleRate() << " Hz, " << m_InputDevice->format().codec().toStdString();
    SetDescription(description.str());

    // status is updated by state-change slot.
  }
  catch (...)
  {
    delete m_InputDevice;
    m_InputDevice = 0;
    SetStatus("Init failed!");
  }
}


//-----------------------------------------------------------------------------
void AudioDataSource::OnReadyRead()
{
  GrabData();
}


//-----------------------------------------------------------------------------
void AudioDataSource::OnStateChanged(QAudio::State state)
{
  switch (state)
  {
    case QAudio::ActiveState:
    case QAudio::IdleState:
      SetStatus("Grabbing");
      break;
    case QAudio::SuspendedState:
    case QAudio::StoppedState:
    default:
      if (m_InputDevice->error() != QAudio::NoError)
      {
        SetStatus("Error");
      }
      else
      {
        SetStatus("Stopped");
      }
      break;
  }
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
  // sanity check
  if (m_InputStream == 0)
    return;

  // beware: m_InputStream->bytesAvailable() always returns zero!
  std::size_t   bytesToRead       = m_InputDevice->bytesReady();
  if (bytesToRead > 0)
  {
    char*         buffer            = new char[bytesToRead];
    std::size_t   bytesActuallyRead = m_InputStream->read(buffer, bytesToRead);
    if (bytesActuallyRead > 0)
    {
      igtl::TimeStamp::Pointer timeCreated = igtl::TimeStamp::New();

      AudioDataType::Pointer wrapper = AudioDataType::New();
      wrapper->SetBlob(buffer, bytesActuallyRead);
      wrapper->SetTimeStampInNanoSeconds(timeCreated->GetTimeInNanoSeconds());
      wrapper->SetDuration(this->m_TimeStampTolerance); // nanoseconds

      AddData(wrapper.GetPointer());
      SetStatus("Grabbing");
    }
  }
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
