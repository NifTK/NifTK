/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkIGITrackerDataSourceService_h
#define niftkIGITrackerDataSourceService_h

#include <niftkIGITrackersExports.h>
#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceLocker.h>
#include <niftkIGILocalDataSourceI.h>
#include <niftkIGITrackerBackend.h>
#include <niftkIGITracker.h>

#include <QObject>
#include <QString>

namespace niftk
{

/**
* \class IGITrackerDataSourceService
* \brief Base class for a simple IGI tracker.
*
* Derived classes must allocate m_Tracker and m_Backend in their
* constructors. All errors should thrown as mitk::Exception
* or sub-classes thereof.
*/
class NIFTKIGITRACKERS_EXPORT IGITrackerDataSourceService : public QObject
                                                          , public IGIDataSource
                                                          , public IGILocalDataSourceI
{

public:

  mitkClassMacroItkParent(IGITrackerDataSourceService, IGIDataSource)
  mitkNewMacro3Param(IGITrackerDataSourceService, QString, QString, mitk::DataStorage::Pointer)

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
  * \see IGIDataSourceI::SetProperties()
  */
  virtual void SetProperties(const IGIDataSourceProperties& properties) override;

  /**
  * \see IGIDataSourceI::GetProperties()
  */
  virtual IGIDataSourceProperties GetProperties() const override;

  /**
  * \see IGIDataSourceI::StopRecording()
  */
  virtual void StopRecording() override;

protected:

  IGITrackerDataSourceService(QString name,
                              QString factoryName,
                              mitk::DataStorage::Pointer dataStorage
                             );
  virtual ~IGITrackerDataSourceService();

  // Used to pass error messages from threads to front end.
  void OnErrorOccurred(QString errorMessage);

  static niftk::IGIDataSourceLocker   s_Lock;
  int                                 m_TrackerNumber;
  niftk::IGITracker::Pointer          m_Tracker;
  niftk::IGITrackerBackend::Pointer   m_BackEnd;

private:

  IGITrackerDataSourceService(const IGITrackerDataSourceService&); // deliberately not implemented
  IGITrackerDataSourceService& operator=(const IGITrackerDataSourceService&); // deliberately not implemented

}; // end class

} // end namespace

#endif
