/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef AudioDataSource_h

#include "niftkIGIGuiExports.h"
#include <DataSources/mitkIGIDataType.h>
#include <DataSources/QmitkIGILocalDataSource.h>
#include <QAudioInput>


// forward-decl
class QAudioDeviceInfo;
class QAudioFormat;
class QFile;


class NIFTKIGIGUI_EXPORT AudioDataType : public mitk::IGIDataType
{
public:
  mitkClassMacro(AudioDataType, mitk::IGIDataType);
  itkNewMacro(AudioDataType);


  void SetBlob(const char* blob, std::size_t length);
  std::pair<const char*, std::size_t> GetBlob() const;

protected:
  AudioDataType();
  virtual ~AudioDataType();


private:
  AudioDataType(const AudioDataType& copyme);
  AudioDataType& operator=(const AudioDataType& assignme);


private:
  const char*   m_AudioBlob;
  std::size_t   m_Length;
};


class NIFTKIGIGUI_EXPORT AudioDataSource : public QmitkIGILocalDataSource
{
  Q_OBJECT

public:
  mitkClassMacro(AudioDataSource, QmitkIGILocalDataSource);
  mitkNewMacro1Param(AudioDataSource, mitk::DataStorage*);


  void SetAudioDevice(QAudioDeviceInfo* device, QAudioFormat* format);



  /** @name Functions from mitk::IGIDataSource */
  //@{
public:
  /** @returns false, always */
  virtual bool GetSaveInBackground() const;

  /** @returns false, always */
  virtual bool ProbeRecordedData(const std::string& path, igtlUint64* firstTimeStampInStore, igtlUint64* lastTimeStampInStore);

  /** @throws std::logic_error, always */
  virtual void StartPlayback(const std::string& path, igtlUint64 firstTimeStamp, igtlUint64 lastTimeStamp);

  /** Does nothing. */
  virtual void StopPlayback();

  /** @throws std::logic_error, always */
  virtual void PlaybackData(igtlUint64 requestedTimeStamp);

  /**
   * @see mitk::IGIDataSource::StartRecording()
   * @see QmitkIGIDataSource::StartRecording()
   */
  virtual void StartRecording(const std::string& directoryPrefix, const bool& saveInBackground, const bool& saveOnReceipt);

  /** @see mitk::IGIDataSource::StartRecording() */
  virtual void StopRecording();


protected:
  /** */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);

  /** @returns true if data is an instance of AudioDataType. */
  virtual bool CanHandleData(mitk::IGIDataType* data) const;

  /** @returns true, always */
  virtual bool Update(mitk::IGIDataType* data);

  //@}


protected:
  /** \see QmitkIGILocalDataSource::GrabData() */
  virtual void GrabData();


protected slots:
  /** @see QAudioInput::stateChanged(QAudio::State) */
  void OnStateChanged(QAudio::State state);

  /** @see QIODevice::readyRead() */
  void OnReadyRead();


protected:
  AudioDataSource(mitk::DataStorage* storage);
  virtual ~AudioDataSource();


  /** @name Not implemented */
  //@{
private:
  AudioDataSource(const AudioDataSource& copyme);
  AudioDataSource& operator=(const AudioDataSource& assignme);
  //@}


  QAudioInput*        m_InputDevice;
  QIODevice*          m_InputStream;      // we do not own this one!

  QFile*              m_OutputFile;
};


#endif // AudioDataSource_h
