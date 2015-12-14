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

#include "niftkNiftyLinkDataType.h"
#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceLocker.h>
#include <niftkIGIWaitForSavedDataSourceBuffer.h>
#include <niftkIGICleanableDataSourceI.h>
#include <niftkIGIDataSourceBackgroundDeleteThread.h>
#include <niftkIGIBufferedSaveableDataSourceI.h>
#include <niftkIGISaveableDataSourceI.h>
#include <niftkIGIDataSourceBackgroundSaveThread.h>
#include <NiftyLinkMessageContainer.h>

#include <igtlTrackingDataMessage.h>
#include <igtlImageMessage.h>
#include <igtlStringMessage.h>
#include <igtlTimeStamp.h>

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
* Also, this class is the common base class of NiftyLinkClientDataSourceService
* and NiftyLinkServerDataSourceService. So, in both cases, there should only be
* 1 connection. So, either this NiftyLinkClientDataSourceService connects to
* 1 external server (e.g. PLUSServer), or 1 external client connects to
* NiftyLinkServerDataSourceService.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NiftyLinkDataSourceService
    : public QObject
    , public IGIDataSource
    , public IGISaveableDataSourceI
    , public IGIBufferedSaveableDataSourceI
    , public IGICleanableDataSourceI
{

  Q_OBJECT

public:

  mitkClassMacroItkParent(NiftyLinkDataSourceService, IGIDataSource);

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
  * \see niftk::IGISaveableDataSourceI::SaveBuffer()
  */
  virtual void SaveBuffer() override;

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  */
  bool ProbeRecordedData(const QString& path,
                         niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                         niftk::IGIDataType::IGITimeType* lastTimeStampInStore) override;

  /**
  * \brief IGIDataSourceI::SetProperties()
  *
  * Also note, that if you set the lag here, it will be applied to all data-sources.
  * So, if you have 1 source eg. PLUS Server, sending tracking and imaging, the remote
  * source needs to ensure that the different types of data have been synchronised in time.
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

  /**
  * \brief Receives ANY NiftyLink message (hence OpenIGTLink) message,
  * and adds its to the buffers. Currently we assume that its tracking data or 2D images.
  * (i.e. relatively small). This design may be inappropriate for large scale (e.g. 4D MR) data.
  */
  void MessageReceived(niftk::NiftyLinkMessageContainer::Pointer message);

private:
  NiftyLinkDataSourceService(const NiftyLinkDataSourceService&); // deliberately not implemented
  NiftyLinkDataSourceService& operator=(const NiftyLinkDataSourceService&); // deliberately not implemented

  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> > GetPlaybackIndex(QString directory);

  std::vector<IGIDataItemInfo> ReceiveTrackingData(QString bufferName,
                                                   niftk::IGIDataType::IGITimeType timeRequested,
                                                   niftk::IGIDataType::IGITimeType actualTime,
                                                   igtl::TrackingDataMessage::Pointer);
  void SaveTrackingData(niftk::NiftyLinkDataType::Pointer, igtl::TrackingDataMessage::Pointer);

  std::vector<IGIDataItemInfo> ReceiveImage(QString bufferName,
                                            niftk::IGIDataType::IGITimeType timeRequested,
                                            niftk::IGIDataType::IGITimeType actualTime,
                                            igtl::ImageMessage::Pointer);
  void SaveImage(niftk::NiftyLinkDataType::Pointer, igtl::ImageMessage::Pointer);

  std::vector<IGIDataItemInfo> ReceiveString(igtl::StringMessage::Pointer);

  void AddAll(const std::vector<IGIDataItemInfo>& a, std::vector<IGIDataItemInfo>& b);

  static niftk::IGIDataSourceLocker                              s_Lock;
  QMutex                                                         m_Lock;
  int                                                            m_SourceNumber;
  niftk::IGIDataType::IGIIndexType                               m_FrameId;
  niftk::IGIDataSourceBackgroundDeleteThread*                    m_BackgroundDeleteThread;
  niftk::IGIDataSourceBackgroundSaveThread*                      m_BackgroundSaveThread;
  int                                                            m_Lag = 0;
  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> >      m_PlaybackIndex;

  // In contrast say to the OpenCV source, we store multiple buffers, key is device name.
  QMap<QString, niftk::IGIWaitForSavedDataSourceBuffer::Pointer> m_Buffers;

  // Make sure this is only used from the MessageReceived thread.
  igtl::TimeStamp::Pointer                                       m_MessageReceivedTimeStamp;

}; // end class

} // end namespace

#endif
