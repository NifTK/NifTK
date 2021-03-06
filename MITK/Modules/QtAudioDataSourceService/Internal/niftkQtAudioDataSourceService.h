/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkQtAudioDataSourceService_h
#define niftkQtAudioDataSourceService_h

#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceLocker.h>

#include <QObject>
#include <QSet>
#include <QMutex>
#include <QString>
#include <QAudioInput>

// forward-decl
class QAudioDeviceInfo;
class QAudioFormat;
class QFile;

namespace niftk
{

/**
* \class QtAudioDataSourceService
* \brief Provides a feed of images from QtAudio, as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class QtAudioDataSourceService
    : public QObject
    , public IGIDataSource
{

  Q_OBJECT

public:

  mitkClassMacroItkParent(QtAudioDataSourceService, IGIDataSource)
  mitkNewMacro3Param(QtAudioDataSourceService, QString, const IGIDataSourceProperties&, mitk::DataStorage::Pointer)

  /**
  * \see  IGIDataSourceI::StartPlayback()
  */
  virtual void StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                             niftk::IGIDataSourceI::IGITimeType lastTimeStamp) override;

  /**
  * \see IGIDataSourceI::PlaybackData()
  */
  void PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp) override;

  /**
  * \see IGIDataSourceI::StopPlayback()
  */
  virtual void StopPlayback() override;

  /**
  * \see IGIDataSourceI::Update()
  */
  virtual std::vector<IGIDataItemInfo> Update(const niftk::IGIDataSourceI::IGITimeType& time) override;

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  */
  bool ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                         niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore) override;

  /**
  * \see IGIDataSourceI::StartRecording()
  */
  virtual void StartRecording() override;

  /**
  * \see IGIDataSourceI::StopRecording()
  */
  virtual void StopRecording() override;

  /**
  * \brief IGIDataSourceI::SetProperties()
  */
  virtual void SetProperties(const IGIDataSourceProperties& properties) override;

  /**
  * \brief IGIDataSourceI::GetProperties()
  */
  virtual IGIDataSourceProperties GetProperties() const override;

protected:

  QtAudioDataSourceService(QString factoryName,
                               const IGIDataSourceProperties& properties,
                               mitk::DataStorage::Pointer dataStorage
                               );
  virtual ~QtAudioDataSourceService();

private slots:

  void OnStateChanged(QAudio::State state);
  void OnReadyRead();

private:

  QtAudioDataSourceService(const QtAudioDataSourceService&); // deliberately not implemented
  QtAudioDataSourceService& operator=(const QtAudioDataSourceService&); // deliberately not implemented

  void StartWAVFile();
  void FinishWAVFile();

  static niftk::IGIDataSourceLocker               s_Lock;
  QMutex                                          m_Lock;
  int                                             m_SourceNumber;
  niftk::IGIDataSourceI::IGIIndexType             m_FrameId;

  QAudioInput*                                    m_InputDevice;
  QIODevice*                                      m_InputStream;      // we do not own this one!
  QFile*                                          m_OutputFile;
  QAudioDeviceInfo                                m_DeviceInfo;
  QAudioFormat                                    m_Inputformat;
  int                                             m_SegmentCounter;

}; // end class

} // end namespace

#endif
