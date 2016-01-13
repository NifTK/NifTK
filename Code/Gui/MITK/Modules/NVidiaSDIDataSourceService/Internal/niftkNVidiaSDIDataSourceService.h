/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkNVidiaSDIDataSourceService_h
#define niftkNVidiaSDIDataSourceService_h

#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceLocker.h>

#include <QObject>
#include <QSet>
#include <QMutex>
#include <QString>

namespace niftk
{

/**
* \class NVidiaSDIDataSourceService
* \brief Provides an NVidia SDI video feed, as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NVidiaSDIDataSourceService
    : public QObject
    , public IGIDataSource    
{

public:

  mitkClassMacroItkParent(NVidiaSDIDataSourceService, IGIDataSource);
  mitkNewMacro3Param(NVidiaSDIDataSourceService, QString, const IGIDataSourceProperties&, mitk::DataStorage::Pointer);

    // This should match libvideo/SDIInput::InterlacedBehaviour
  enum InterlacedBehaviour
  {
    DO_NOTHING_SPECIAL    = 0,
    DROP_ONE_FIELD        = 1,
    STACK_FIELDS          = 2, /** No longer supported! */
    SPLIT_LINE_INTERLEAVED_STEREO = 3
  };

  /**
  * \see  IGIDataSourceI::StartPlayback()
  */
  virtual void StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                             niftk::IGIDataType::IGITimeType lastTimeStamp) override;

  /**
  * \see IGIDataSourceI::PlaybackData()
  */
  void PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp) override;

  /**
  * \see IGIDataSourceI::StopPlayback()
  */
  virtual void StopPlayback() override;

  /**
  * \see IGIDataSourceI::GetRecordingDirectoryName()
  */
  virtual QString GetRecordingDirectoryName() override;

  /**
  * \see IGIDataSourceI::Update()
  */
  virtual std::vector<IGIDataItemInfo> Update(const niftk::IGIDataType::IGITimeType& time) override;

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  */
  bool ProbeRecordedData(const QString& path,
                         niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                         niftk::IGIDataType::IGITimeType* lastTimeStampInStore) override;

  /**
  * \brief IGIDataSourceI::SetProperties()
  */
  virtual void SetProperties(const IGIDataSourceProperties& properties) override;

  /**
  * \brief IGIDataSourceI::GetProperties()
  */
  virtual IGIDataSourceProperties GetProperties() const override;

protected:

  NVidiaSDIDataSourceService(QString factoryName,
                               const IGIDataSourceProperties& properties,
                               mitk::DataStorage::Pointer dataStorage
                               );
  virtual ~NVidiaSDIDataSourceService();

private:

  NVidiaSDIDataSourceService(const NVidiaSDIDataSourceService&); // deliberately not implemented
  NVidiaSDIDataSourceService& operator=(const NVidiaSDIDataSourceService&); // deliberately not implemented

  static niftk::IGIDataSourceLocker               s_Lock;
  QMutex                                          m_Lock;
  int                                             m_ChannelNumber;
  niftk::IGIDataType::IGIIndexType                m_FrameId;

}; // end class

} // end namespace

#endif
