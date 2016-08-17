/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkQtCameraVideoDataSourceService_h
#define niftkQtCameraVideoDataSourceService_h

#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceLocker.h>
#include <niftkIGIDataSourceBuffer.h>
#include <niftkIGILocalDataSourceI.h>
#include <niftkIGIDataSourceGrabbingThread.h>
#include <niftkIGICleanableDataSourceI.h>
#include <niftkIGIDataSourceBackgroundDeleteThread.h>
#include <niftkIGIBufferedSaveableDataSourceI.h>

#include <QObject>
#include <QSet>
#include <QMutex>
#include <QString>

namespace niftk
{

/**
* \class QtCameraVideoDataSourceService
* \brief Provides an QtCamera video feed, as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class QtCameraVideoDataSourceService
    : public QObject
    , public IGIDataSource
    , public IGILocalDataSourceI
    , public IGICleanableDataSourceI
    , public IGIBufferedSaveableDataSourceI
{

public:

  mitkClassMacroItkParent(QtCameraVideoDataSourceService, IGIDataSource)
  mitkNewMacro3Param(QtCameraVideoDataSourceService, QString, const IGIDataSourceProperties&, mitk::DataStorage::Pointer)

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
  * \see IGIDataSourceI::Update()
  */
  virtual std::vector<IGIDataItemInfo> Update(const niftk::IGIDataType::IGITimeType& time) override;

  /**
  * \see niftk::IGIDataSource::CleanBuffer()
  */
  virtual void CleanBuffer() override;

  /**
  * \see niftk::IGILocalDataSourceI::GrabData()
  */
  virtual void GrabData() override;

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  */
  bool ProbeRecordedData(niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
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

  QtCameraVideoDataSourceService(QString factoryName,
                               const IGIDataSourceProperties& properties,
                               mitk::DataStorage::Pointer dataStorage
                               );
  virtual ~QtCameraVideoDataSourceService();

private:

  QtCameraVideoDataSourceService(const QtCameraVideoDataSourceService&); // deliberately not implemented
  QtCameraVideoDataSourceService& operator=(const QtCameraVideoDataSourceService&); // deliberately not implemented

  void SaveItem(niftk::IGIDataType::Pointer item) override;

  static niftk::IGIDataSourceLocker               s_Lock;
  QMutex                                          m_Lock;
  int                                             m_ChannelNumber;
  niftk::IGIDataType::IGIIndexType                m_FrameId;
  niftk::IGIDataSourceBuffer::Pointer             m_Buffer;
  niftk::IGIDataSourceBackgroundDeleteThread*     m_BackgroundDeleteThread;
  niftk::IGIDataSourceGrabbingThread*             m_DataGrabbingThread;
  std::set<niftk::IGIDataType::IGITimeType>       m_PlaybackIndex;

}; // end class

} // end namespace

#endif
