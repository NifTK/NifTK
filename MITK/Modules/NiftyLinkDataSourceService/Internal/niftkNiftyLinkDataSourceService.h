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
#include <niftkIGIDataSourceWaitingBuffer.h>
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
* NiftyLink client and server classes are already threaded, so messages
* simply appear via a Qt signal. This class therefore registers to this
* signal to receive new messages.
*
* This class is the common base class of NiftyLinkClientDataSourceService
* and NiftyLinkServerDataSourceService. So, in both cases, there should only be
* 1 connection. So, either NiftyLinkClientDataSourceService connects to
* 1 external server (e.g. PLUSServer), or 1 external client connects to
* NiftyLinkServerDataSourceService. If you run this class as a
* NiftyLinkServerDataSourceService, and multiple clients try to connect,
* then each client should be setting a unique device name on their messages.
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

  mitkClassMacroItkParent(NiftyLinkDataSourceService, IGIDataSource)

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
  * \see niftk::IGIDataSource::SaveItem()
  */
  virtual void SaveItem(niftk::IGIDataType& item) override;

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
  bool ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                         niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore) override;

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

  std::vector<IGIDataItemInfo> ReceiveTrackingData(std::string bufferName,
                                                   niftk::IGIDataSourceI::IGITimeType timeRequested,
                                                   niftk::IGIDataSourceI::IGITimeType actualTime,
                                                   igtl::TrackingDataMessage*);

  void SaveTrackingData(niftk::NiftyLinkDataType&, igtl::TrackingDataMessage*);
  void LoadTrackingData(const niftk::IGIDataSourceI::IGITimeType& actualTime, QStringList& listOfFileNames);

  std::vector<IGIDataItemInfo> ReceiveImage(std::string bufferName,
                                            niftk::IGIDataSourceI::IGITimeType timeRequested,
                                            niftk::IGIDataSourceI::IGITimeType actualTime,
                                            igtl::ImageMessage*);

  void SaveImage(niftk::NiftyLinkDataType&, igtl::ImageMessage*);
  void LoadImage(const niftk::IGIDataSourceI::IGITimeType& actualTime, QStringList& listOfFileNames);

  std::vector<IGIDataItemInfo> ReceiveString(igtl::StringMessage*);

  void AddAll(const std::vector<IGIDataItemInfo>& a, std::vector<IGIDataItemInfo>& b);
  QString GetDirectoryNamePart(const QString& fullPathName, int indexFromEnd);

  QMutex                                                                     m_Lock;
  niftk::IGIDataSourceI::IGIIndexType                                        m_FrameId;
  niftk::IGIDataSourceBackgroundDeleteThread*                                m_BackgroundDeleteThread;
  niftk::IGIDataSourceBackgroundSaveThread*                                  m_BackgroundSaveThread;
  int                                                                        m_Lag;

  // In contrast say to the OpenCV source, we store multiple playback indexes, key is device name.
  QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> >               m_PlaybackIndex;
  QMap<QString, QHash<niftk::IGIDataSourceI::IGITimeType, QStringList> >     m_PlaybackFiles;

  // In contrast say to the OpenCV source, we store multiple buffers, key is device name.
  std::map<std::string, std::unique_ptr<niftk::IGIDataSourceWaitingBuffer> > m_Buffers;

  // Make sure this is only used from the MessageReceived thread.
  igtl::TimeStamp::Pointer                                                   m_MessageCreatedTimeStamp;
  niftk::NiftyLinkDataType                                                   m_CachedDataType;

}; // end class

} // end namespace

#endif
