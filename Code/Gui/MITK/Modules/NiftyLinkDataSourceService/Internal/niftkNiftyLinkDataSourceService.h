/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkNiftyLinkDataSourceService_h
#define niftkNiftyLinkDataSourceService_h

#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceLocker.h>
#include <niftkIGIDataSourceBuffer.h>
#include <niftkIGIDataSourceBackgroundDeleteThread.h>
#include <niftkIGIDataSourceGrabbingThread.h>
#include <niftkIGILocalDataSourceI.h>
#include <NiftyLinkMessageContainer.h>

#include <QObject>
#include <QSet>
#include <QMutex>
#include <QString>
#include <QAbstractSocket>

namespace niftk
{

/**
* \class NiftyLinkDataSourceService
* \brief Abstract base class for both NiftyLink Client and Server sources.
*
* In contrast say to niftk::OpenCVVideoDataSourceService, or
* niftk::MITKTrackerDataSourceService which directly grab data in
* a specific data grabbing thread, here, data comes via a socket.
* So, the NiftyLink client and server classes are already threaded.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NiftyLinkDataSourceService
    : public QObject
    , public IGIDataSource
    , public IGILocalDataSourceI
{

  Q_OBJECT

public:

  mitkClassMacroItkParent(NiftyLinkDataSourceService, IGIDataSource);

  /**
  * \see IGIDataSourceI::StartCapturing()
  */
  virtual void StartCapturing() override;

  /**
  * \see IGIDataSourceI::StopCapturing()
  */
  virtual void StopCapturing() override;

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
  * \see niftk::IGIDataSource::SaveItem()
  */
  virtual void SaveItem(niftk::IGIDataType::Pointer item) override;

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

  NiftyLinkDataSourceService(QString name,
                             QString factoryName,
                             const IGIDataSourceProperties& properties,
                             mitk::DataStorage::Pointer dataStorage
                             );
  virtual ~NiftyLinkDataSourceService();

private:
  NiftyLinkDataSourceService(const NiftyLinkDataSourceService&); // deliberately not implemented
  NiftyLinkDataSourceService& operator=(const NiftyLinkDataSourceService&); // deliberately not implemented

  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> > GetPlaybackIndex(QString directory);

  static niftk::IGIDataSourceLocker                         s_Lock;
  QMutex                                                    m_Lock;
  int                                                       m_SourceNumber;
  niftk::IGIDataType::IGIIndexType                          m_FrameId;
  niftk::IGIDataSourceBackgroundDeleteThread*               m_BackgroundDeleteThread;
  niftk::IGIDataSourceGrabbingThread*                       m_DataGrabbingThread;
  int                                                       m_Lag = 0;
  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> > m_PlaybackIndex;

  // In contrast say to the OpenCV source, we store multiple buffers, keyed by tool name.
  QMap<QString, niftk::IGIDataSourceBuffer::Pointer>        m_Buffers;

}; // end class

} // end namespace

#endif
