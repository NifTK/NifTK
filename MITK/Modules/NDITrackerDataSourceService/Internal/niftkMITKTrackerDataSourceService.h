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
#include <niftkIGIDataSourceLocker.h>
#include <niftkIGIDataSourceRingBuffer.h>
#include <niftkIGILocalDataSourceI.h>
#include <niftkIGIDataSourceGrabbingThread.h>
#include "niftkIGITrackerDataType.h"

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
    : public QObject
    , public IGIDataSource
    , public IGILocalDataSourceI
{

public:

  mitkClassMacroItkParent(MITKTrackerDataSourceService, IGIDataSource)
  mitkNewMacro5Param(MITKTrackerDataSourceService, QString, QString,
                     const IGIDataSourceProperties&, mitk::DataStorage::Pointer, niftk::NDITracker::Pointer)

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
  * \see niftk::IGILocalDataSourceI::GrabData()
  */
  virtual void GrabData() override;

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  */
  bool ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                         niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore) override;

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

  QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> > GetPlaybackIndex(QString directory);

  void SaveItem(const std::unique_ptr<niftk::IGIDataType>& item);

  static niftk::IGIDataSourceLocker                                       s_Lock;
  QMutex                                                                  m_Lock;
  int                                                                     m_TrackerNumber;
  int                                                                     m_Lag;
  niftk::IGIDataSourceI::IGIIndexType                                     m_FrameId;
  niftk::IGIDataSourceGrabbingThread*                                     m_DataGrabbingThread;
  QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> >            m_PlaybackIndex;

  // The main tracker.
  niftk::NDITracker::Pointer                                              m_Tracker;

  // In contrast say to the OpenCV source, we store multiple buffers, keyed by tool name.
  std::map<std::string, std::unique_ptr<niftk::IGIDataSourceRingBuffer> > m_Buffers;
  niftk::IGITrackerDataType                                               m_CachedDataType;

}; // end class

} // end namespace

#endif
