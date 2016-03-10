/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtAudioDataSourceService.h"
#include "niftkQtAudioDataType.h"
#include <niftkIGIDataSourceUtils.h>
#include <niftkIGIDataSourceI.h>
#include <mitkExceptionMacro.h>
#include <QDir>
#include <QMutexLocker>
#include <QAudioFormat>
#include <QAudioDeviceInfo>
#include <QAudioInput>
#include <QThread>

Q_DECLARE_METATYPE(QAudioFormat);

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker QtAudioDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
QtAudioDataSourceService::QtAudioDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("QtAudio-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_FrameId(0)
, m_InputDevice(NULL)
, m_InputStream(NULL)
, m_OutputFile(NULL)
{
  if(!properties.contains("name"))
  {
    mitkThrow() << "Audio device name not specified!";
  }
  QString audioDeviceName = (properties.value("name")).toString();

  if (!properties.contains("format"))
  {
    mitkThrow() << "Audio format not specified!";
  }
  QAudioFormat audioDeviceFormat = properties.value("format").value<QAudioFormat>();

  if (!properties.contains("formatString"))
  {
    mitkThrow() << "Format string not specified!";
  }
  QString formatString = properties.value("formatString").toString();

  QString foundName;
  QAudioDeviceInfo audioDeviceInfo;

  QList<QAudioDeviceInfo> allDevices = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
  foreach(QAudioDeviceInfo d, allDevices)
  {
    if (d.deviceName() ==  audioDeviceName)
    {
      foundName = d.deviceName();
      audioDeviceInfo = d;
    }
  }
  if (foundName.isEmpty())
  {
    mitkThrow() << "Audio format not supported!";
  }

  this->SetStatus("Initialising");

  QString deviceName = this->GetName();
  m_SourceNumber = (deviceName.remove(0, 8)).toInt(); // Should match string QtAudio- above

  m_InputDevice = new QAudioInput(audioDeviceInfo, audioDeviceFormat);
  bool ok = false;
  ok = QObject::connect(m_InputDevice, SIGNAL(stateChanged(QAudio::State)), this, SLOT(OnStateChanged(QAudio::State)));
  assert(ok);

  m_DeviceInfo  = audioDeviceInfo;
  m_Inputformat = audioDeviceFormat;

  this->SetTimeStampTolerance(40*1000000);
  this->SetDescription(formatString);
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
QtAudioDataSourceService::~QtAudioDataSourceService()
{
  // sanity check: dont expect any background thread.
  assert(this->thread() == QThread::currentThread());

  bool    ok = false;

  if (m_InputDevice != 0)
  {
    m_InputDevice->stop();

    ok = QObject::disconnect(m_InputDevice, SIGNAL(stateChanged(QAudio::State)), this, SLOT(OnStateChanged(QAudio::State)));
    assert(ok);

    delete m_InputDevice;
    m_InputDevice = 0;
  }

  s_Lock.RemoveSource(m_SourceNumber);
  delete m_OutputFile;
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::OnStateChanged(QAudio::State state)
{
  switch (state)
  {
    case QAudio::ActiveState:
      this->SetStatus("Active");
      break;
    case QAudio::IdleState:
      this->SetStatus("Idle");
      break;
    case QAudio::SuspendedState:
      this->SetStatus("Suspended");
      break;
    case QAudio::StoppedState:
      this->SetStatus("Stopped");
      break;
    default:
      if (m_InputDevice->error() != QAudio::NoError)
      {
        this->SetStatus("Error");
        MITK_INFO << "QtAudioDataSourceService::OnStateChanged(Error):" << m_InputDevice->error();
      }
      else
      {
        SetStatus("OK");
      }
      break;
  }
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::OnReadyRead()
{
  // sanity check: dont expect any background thread.
  assert(this->thread() == QThread::currentThread());

  // sanity check
  if (m_InputStream == 0)
  {
    mitkThrow() << "Invalid output stream!";
  }
  if (m_InputDevice == 0)
  {
    mitkThrow() << "Invalid audio device!";
  }

  // sanity check: should only be called while recording
  if (!this->GetIsRecording())
  {
    mitkThrow() << "Audio recording only";
  }

  // beware: m_InputStream->bytesAvailable() always returns zero!
  std::size_t bytesToRead = m_InputDevice->bytesReady();
  if (bytesToRead > 0)
  {
    char*         buffer            = new char[bytesToRead];
    std::size_t   bytesActuallyRead = m_InputStream->read(buffer, bytesToRead);

    if (bytesActuallyRead > 0)
    {
      // if writing the current blob to the file would make it too big (signed 32 bit overflow!),
      // then start a new segment.
      // this can happen quite easily: 32 bit samples, 2 channels, 96kHz --> 46 minutes!
      if ((m_OutputFile->size() + bytesActuallyRead) > std::numeric_limits<int>::max())
      {
        FinishWAVFile();
        ++m_SegmentCounter;
        StartWAVFile();
      }

      std::size_t actuallyWritten = m_OutputFile->write(buffer, bytesActuallyRead);
      assert(actuallyWritten == bytesActuallyRead);

    }
    delete buffer;
  }
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::StartWAVFile()
{
  QString directoryPath = this->GetRecordingDirectory();
  QDir directory(directoryPath);
  if (directory.mkpath(directoryPath))
  {
    assert(m_SegmentCounter > 0);
    m_OutputFile = new QFile(directory.absoluteFilePath(QString("%1.wav").arg(m_SegmentCounter)));

    // it's an error to overwrite a file!
    // first, each segment should have a unique number.
    // second, each datasource recording session goes into its own directory.
    bool ok = !m_OutputFile->exists();
    if (ok)
    {
      ok = m_OutputFile->open(QIODevice::WriteOnly);
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
    }

    if (!ok)
    {
      m_InputDevice->stop();
      this->SetStatus("Error: cannot open output file");
    }
  }
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::FinishWAVFile()
{
  // fill in the missing chunk sizes
  unsigned int riffsize = m_OutputFile->size() - 8;
  m_OutputFile->seek(4);
  m_OutputFile->write((const char*) &riffsize, sizeof(riffsize));

  unsigned int datasize = m_OutputFile->size() - 44;
  m_OutputFile->seek(40);
  m_OutputFile->write((const char*) &datasize, sizeof(datasize));

  m_OutputFile->flush();
  delete m_OutputFile;
  m_OutputFile = 0;
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  MITK_INFO << "QtAudioDataSourceService::SetProperties(), nothing to do";
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties QtAudioDataSourceService::GetProperties() const
{
  MITK_INFO << "QtAudioDataSourceService::GetProperties(), nothing to return";
  IGIDataSourceProperties props;
  return props;
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                                             niftk::IGIDataType::IGITimeType lastTimeStamp)
{
  mitkThrow() << "QtAudioDataSourceService::StartPlayback(), Not implemented yet!";
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::StopPlayback()
{
  mitkThrow() << "QtAudioDataSourceService::StopPlayback(), Not implemented yet!";
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp)
{
  mitkThrow() << "QtAudioDataSourceService::PlaybackData(), Not implemented yet!";
}


//-----------------------------------------------------------------------------
bool QtAudioDataSourceService::ProbeRecordedData(niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                                                 niftk::IGIDataType::IGITimeType* lastTimeStampInStore)
{  
  return false;
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::StartRecording()
{
  assert(m_OutputFile == 0);

  IGIDataSource::StartRecording();

  // each recording session starts counting its own segments.
  m_SegmentCounter = 1;

  m_InputDevice->setBufferSize(1024*1024*500); // 5Mb
  m_InputStream = m_InputDevice->start();

  bool ok = false;
  ok = QObject::connect(m_InputStream, SIGNAL(readyRead()), this, SLOT(OnReadyRead()));
  assert(ok);

  StartWAVFile();
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceService::StopRecording()
{
  assert(m_OutputFile != 0);

  FinishWAVFile();

  bool ok = false;
  ok = QObject::disconnect(m_InputStream, SIGNAL(readyRead()), this, SLOT(OnReadyRead()));
  assert(ok);

  m_InputDevice->stop();

  // reset to invalid
  m_SegmentCounter = 0;

  IGIDataSource::StopRecording();
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> QtAudioDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;
  IGIDataItemInfo info;
  info.m_Name = this->GetName();
  info.m_IsLate = false;
  info.m_LagInMilliseconds = 0;
  info.m_FramesPerSecond = 0;
  infos.push_back(info);
  return infos;
}

} // end namespace
