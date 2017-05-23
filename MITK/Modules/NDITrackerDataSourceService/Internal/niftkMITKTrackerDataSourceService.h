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
#include <niftkIGILocalDataSourceI.h>
#include <niftkIGIDataSourceGrabbingThread.h>
#include <niftkIGITrackerBackend.h>
#include <niftkNDITracker.h>

#include <QObject>
#include <QString>

namespace niftk
{

/**
* \class MITKTrackerDataSourceService
* \brief Provides a local MITK implementation of a tracker interface,
* as an IGIDataSourceServiceI. The other class niftk::NDITracker provides
* the main tracking mechanism, utilising PLUS/Atami to speak to the serial port
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
  virtual void PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp) override;

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

  static niftk::IGIDataSourceLocker   s_Lock;
  int                                 m_TrackerNumber;
  niftk::IGIDataSourceGrabbingThread* m_DataGrabbingThread;
  niftk::NDITracker::Pointer          m_Tracker;
  niftk::IGITrackerBackend::Pointer   m_BackEnd;

}; // end class

} // end namespace

#endif
