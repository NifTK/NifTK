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
#include <sstream>
#include <QAudioInput>
#include <QAudioDeviceInfo>
#include <QFile>
#include <QDir>


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
std::pair<const char*, std::size_t> AudioDataType::GetBlob() const
{
  return std::make_pair(m_AudioBlob, m_Length);
}


//-----------------------------------------------------------------------------
AudioDataSource::AudioDataSource(mitk::DataStorage* storage)
  : QmitkIGILocalDataSource(storage)
  , m_InputDevice(0)
  , m_OutputFile(0)
  , m_DeviceInfo(0)
  , m_Inputformat(0)
{
  SetStatus("Initialising...");

  QAudioDeviceInfo  defaultDevice = QAudioDeviceInfo::defaultInputDevice();
  QAudioFormat      defaultFormat = defaultDevice.preferredFormat();
  // try not to do 8 bit, sounds like trash.
  defaultFormat.setSampleSize(16);
  if (!defaultDevice.isFormatSupported(defaultFormat))
    // nearestFormat() is a bit stupid: say we request unsupported 32 bit, it will pick 8 instead of 16 bit.
    defaultFormat = defaultDevice.nearestFormat(defaultFormat);

  SetAudioDevice(&defaultDevice, &defaultFormat);
}


//-----------------------------------------------------------------------------
AudioDataSource::~AudioDataSource()
{
  DisconnectAudio();

  delete m_OutputFile;
}


//-----------------------------------------------------------------------------
const QAudioDeviceInfo* AudioDataSource::GetDeviceInfo() const
{
  return m_DeviceInfo;
}


//-----------------------------------------------------------------------------
const QAudioFormat* AudioDataSource::GetFormat() const
{
  return m_Inputformat;
}


//-----------------------------------------------------------------------------
void AudioDataSource::DisconnectAudio()
{
  bool    ok = false;

  if (m_InputDevice != 0)
  {
    if (m_InputStream != 0)
    {
      ok = QObject::disconnect(m_InputStream, SIGNAL(readyRead()), this, SLOT(OnReadyRead()));
      assert(ok);
      // we do not own m_InputStream!
      m_InputStream = 0;
    }
    m_InputDevice->stop();

    ok = QObject::disconnect(m_InputDevice, SIGNAL(stateChanged(QAudio::State)), this, SLOT(OnStateChanged(QAudio::State)));
    assert(ok);

    delete m_InputDevice;
    m_InputDevice = 0;
  }

  delete m_DeviceInfo;
  delete m_Inputformat;
}


//-----------------------------------------------------------------------------
void AudioDataSource::SetAudioDevice(QAudioDeviceInfo* device, QAudioFormat* format)
{
  assert(device != 0);
  assert(format != 0);

  DisconnectAudio();

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
    description << formatToString(format).toStdString();
    SetDescription(description.str());

    // status is updated by state-change slot.

    m_DeviceInfo  = new QAudioDeviceInfo(*device);
    m_Inputformat = new QAudioFormat(*format);
  }
  catch (...)
  {
    delete m_InputDevice;
    m_InputDevice = 0;
    SetStatus("Init failed!");
  }
}


//-----------------------------------------------------------------------------
QString AudioDataSource::formatToString(const QAudioFormat* format)
{
  // FIXME: sample type?
  return QString("%1 channels @ %2 Hz, %3 bit, %4").arg(format->channels()).arg(format->sampleRate()).arg(format->sampleSize()).arg(format->codec());
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
  return dynamic_cast<AudioDataType*>(data) != 0;
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
  if (m_InputDevice == 0)
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
void AudioDataSource::StartRecording(const std::string& directoryPrefix, const bool& saveInBackground, const bool& saveOnReceipt)
{
  // sanity check
  assert(m_OutputFile == 0);

  // base-class. whatever it does...
  QmitkIGILocalDataSource::StartRecording(directoryPrefix, saveInBackground, saveOnReceipt);


  std::string   directoryPath = GetSaveDirectoryName();
  QDir          directory(QString::fromStdString(directoryPath));
  if (directory.mkpath(QString::fromStdString(directoryPath)))
  {
    m_OutputFile = new QFile(directory.absoluteFilePath("1.wav"));
    bool ok = m_OutputFile->open(QIODevice::WriteOnly);
    if (ok)
    {
      // basic wave header
      // https://ccrma.stanford.edu/courses/422/projects/WaveFormat/
      char   wavheader[44];
      std::memset(&wavheader[0], 0, sizeof(wavheader));
      wavheader[0] = 'R'; wavheader[1] = 'I'; wavheader[2] = 'F'; wavheader[3] = 'F';
      // followed by file size minus 8
      wavheader[8] = 'W'; wavheader[9] = 'A'; wavheader[10] = 'V'; wavheader[11] = 'E';
      wavheader[12] = 'f'; wavheader[13] = 'm'; wavheader[14] = 't'; wavheader[15] = ' ';
      *((unsigned int*  ) &wavheader[16]) = 16;   // fixed size fmt chunk
      *((unsigned short*) &wavheader[20]) = 1;   // pcm
      *((unsigned short*) &wavheader[22]) = m_InputDevice->format().channels();
      *((unsigned int*  ) &wavheader[24]) = m_InputDevice->format().sampleRate();
      *((unsigned int*  ) &wavheader[28]) = m_InputDevice->format().sampleRate() * m_InputDevice->format().channels() * m_InputDevice->format().sampleSize() / 8;
      *((unsigned short*) &wavheader[32]) = m_InputDevice->format().channels() * m_InputDevice->format().sampleSize() / 8;
      *((unsigned short*) &wavheader[34]) = m_InputDevice->format().sampleSize();
      wavheader[36] = 'd'; wavheader[37] = 'a'; wavheader[38] = 't'; wavheader[39] = 'a';
      // followed by data size (filesize minus 44)

      std::size_t actuallyWritten = m_OutputFile->write(&wavheader[0], sizeof(wavheader));
      assert(actuallyWritten == sizeof(wavheader));
      // and after that raw data.
    }
    else
    {
      m_InputDevice->stop();
      SetStatus("Error: cannot open output file");
    }
  }
}


//-----------------------------------------------------------------------------
void AudioDataSource::StopRecording()
{
  assert(m_OutputFile != 0);

  QmitkIGILocalDataSource::StopRecording();

  // fill in the missing chunk sizes
  unsigned int    riffsize = m_OutputFile->size() - 8;
  m_OutputFile->seek(4);
  m_OutputFile->write((const char*) &riffsize, sizeof(riffsize));

  unsigned int    datasize = m_OutputFile->size() - 44;
  m_OutputFile->seek(40);
  m_OutputFile->write((const char*) &datasize, sizeof(datasize));


  m_OutputFile->flush();
  delete m_OutputFile;
  m_OutputFile = 0;
}


//-----------------------------------------------------------------------------
bool AudioDataSource::SaveData(mitk::IGIDataType* d, std::string& outputFileName)
{
  // cannot record while playing back
  assert(!GetIsPlayingBack());

  if (m_OutputFile == 0)
    return false;

  // keep the packet alive.
  AudioDataType::Pointer    data = dynamic_cast<AudioDataType*>(d);
  if (data.IsNull())
    return false;

  std::pair<const char*, std::size_t>   blob = data->GetBlob();
  std::size_t actuallyWritten = m_OutputFile->write(blob.first, blob.second);
  assert(actuallyWritten == blob.second);

  outputFileName = m_OutputFile->fileName().toStdString();

  return true;
}


//-----------------------------------------------------------------------------
bool AudioDataSource::Update(mitk::IGIDataType* data)
{
  return true;
}
