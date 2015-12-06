/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkMITKTrackerDataSourceService_h
#define niftkMITKTrackerDataSourceService_h

#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceBuffer.h>
#include <niftkIGIDataSourceBackgroundDeleteThread.h>
#include <niftkIGIDataSourceGrabbingThread.h>
#include <niftkIGILocalDataSourceI.h>
#include <niftkNDITracker.h>

#include <QObject>
#include <QSet>
#include <QMutex>
#include <QString>

namespace niftk
{

/**
* \class MITKTrackerDataSourceService
* \brief Provides a local MITK implementation of a tracker interface,
* as an IGIDataSourceServiceI. The other class niftk::NDITracker provides
* the main tracking mechanism, utilising MITK to speak to the serial port
* and grab data etc. This class therefore is to coordinate threads, buffers, etc.
* and to function as a MicroService.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class MITKTrackerDataSourceService
    : public IGIDataSource
    , public IGILocalDataSourceI
    , public QObject
{

public:

  mitkClassMacroItkParent(MITKTrackerDataSourceService, IGIDataSource);
  mitkNewMacro5Param(MITKTrackerDataSourceService, QString, QString, const IGIDataSourceProperties&, mitk::DataStorage::Pointer, niftk::NDITracker::Pointer);

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

  MITKTrackerDataSourceService(QString name,
                               QString factoryName,
                               const IGIDataSourceProperties& properties,
                               mitk::DataStorage::Pointer dataStorage,
                               niftk::NDITracker::Pointer tracker
                              );
  virtual ~MITKTrackerDataSourceService();

private:

  MITKTrackerDataSourceService(const MITKTrackerDataSourceService&); // deliberately not implemented
  MITKTrackerDataSourceService& operator=(const MITKTrackerDataSourceService&); // deliberately not implemented

  static int GetNextTrackerNumber();

  static QMutex                                             s_Lock;
  static QSet<int>                                          s_SourcesInUse;

  QMutex                                                    m_Lock;
  int                                                       m_TrackerNumber;
  niftk::IGIDataType::IGIIndexType                          m_FrameId;
  niftk::IGIDataSourceBackgroundDeleteThread*               m_BackgroundDeleteThread;
  niftk::IGIDataSourceGrabbingThread*                       m_DataGrabbingThread;
  int                                                       m_Lag = 0;
  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> > m_PlaybackIndex;

  // The main tracker.
  niftk::NDITracker::Pointer                                m_Tracker;

  // In contrast say to the OpenCV source, we store multiple buffers, keyed by tool name.
  QMap<QString, niftk::IGIDataSourceBuffer::Pointer>        m_Buffers;

}; // end class

} // end namespace

#endif
