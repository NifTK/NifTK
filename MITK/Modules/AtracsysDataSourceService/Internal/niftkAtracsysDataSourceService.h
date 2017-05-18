/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkAtracsysDataSourceService_h
#define niftkAtracsysDataSourceService_h

#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceLocker.h>

#include <QObject>

namespace niftk
{

/**
* \class AtracsysDataSourceService
* \brief Provides an interface to an Atracsys Fusion Track 500,
* as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class AtracsysDataSourceService : public QObject
                                , public IGIDataSource
{

  Q_OBJECT

public:

  mitkClassMacroItkParent(AtracsysDataSourceService,
                          IGIDataSource)

  mitkNewMacro3Param(AtracsysDataSourceService, QString,
                     const IGIDataSourceProperties&, mitk::DataStorage::Pointer)

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

  AtracsysDataSourceService(QString factoryName,
                            const IGIDataSourceProperties& properties,
                            mitk::DataStorage::Pointer dataStorage
                           );

  virtual ~AtracsysDataSourceService();

  static const int ATRACSYS_FRAMES_PER_SECOND;
  static const int ATRACSYS_TIMEOUT;

private slots:

private:

  AtracsysDataSourceService(const AtracsysDataSourceService&); // deliberately not implemented
  AtracsysDataSourceService& operator=(const AtracsysDataSourceService&); // deliberately not implemented

  static niftk::IGIDataSourceLocker s_Lock;
  int                               m_Lag;

}; // end class

} // end namespace

#endif
