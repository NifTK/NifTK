/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKTrackerDataSourceService.h"
#include <niftkIGIMatrixPerFileBackend.h>
#include <niftkCoordinateAxesData.h>

#include <mitkExceptionMacro.h>

#include <QDir>
#include <QMutexLocker>

#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceLocker MITKTrackerDataSourceService::s_Lock;


//-----------------------------------------------------------------------------
MITKTrackerDataSourceService::MITKTrackerDataSourceService(
  QString name,
  QString factoryName,
  const IGIDataSourceProperties& properties,
  mitk::DataStorage::Pointer dataStorage,
  niftk::NDITracker::Pointer tracker
)
: IGIDataSource((name + QString("-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_DataGrabbingThread(NULL)
, m_Tracker(tracker)
{
  if (m_Tracker.IsNull())
  {
    mitkThrow() << "Tracker is NULL!";
  }

  this->SetStatus("Initialising");

  QString deviceName = this->GetName();
  m_TrackerNumber = (deviceName.remove(0, name.length() + 1)).toInt();

  m_BackEnd = niftk::IGIMatrixPerFileBackend::New(this->GetName(), this->GetDataStorage());
  m_BackEnd->SetExpectedFramesPerSecond(m_Tracker->GetExpectedFramesPerSecond());

  this->SetTimeStampTolerance((m_Tracker->GetExpectedFramesPerSecond()
                               / m_Tracker->GetExpectedNumberOfTools())
                              *1000000 // convert to nanoseconds
                              *5       // allow up to 5 frames late
                             );

  this->SetProperties(properties);
  this->SetShouldUpdate(true);

  m_DataGrabbingThread = new niftk::IGIDataSourceGrabbingThread(NULL, this);
  m_DataGrabbingThread->SetInterval(1000 / m_Tracker->GetExpectedFramesPerSecond());
  m_DataGrabbingThread->start();

  if (!m_DataGrabbingThread->isRunning())
  {
    mitkThrow() << "Failed to start data grabbing thread";
  }

  this->SetDescription("MITK Tracker:" + this->GetName());
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
MITKTrackerDataSourceService::~MITKTrackerDataSourceService()
{
  m_DataGrabbingThread->ForciblyStop();
  delete m_DataGrabbingThread;

  s_Lock.RemoveSource(m_TrackerNumber);
}


//-----------------------------------------------------------------------------
bool MITKTrackerDataSourceService::ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                                     niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  return m_BackEnd->ProbeRecordedData(this->GetPlaybackDirectory(),
                                      firstTimeStampInStore,
                                      lastTimeStampInStore);
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataSourceI::IGITimeType lastTimeStamp)
{
  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);
  m_BackEnd->StartPlayback(this->GetPlaybackDirectory(), firstTimeStamp, lastTimeStamp);
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::StopPlayback()
{
  IGIDataSource::StopPlayback();
  m_BackEnd->StopPlayback();
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp)
{
  assert(this->GetIsPlayingBack());

  m_BackEnd->PlaybackData(this->GetPlaybackDirectory(),
                          this->GetTimeStampTolerance(),
                          requestedTimeStamp);

  this->SetStatus("Playing back");
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::GrabData()
{
  if (this->GetIsPlayingBack())
  {
    return;
  }

  if (m_Tracker.IsNull())
  {
    mitkThrow() << "Tracker is null. This should not happen! It's a programming bug.";
  }

  if (m_BackEnd.IsNull())
  {
    mitkThrow() << "Backend is null. This should not happen! It's a programming bug.";
  }

  niftk::IGIDataSourceI::IGITimeType timeCreated = this->GetTimeStampInNanoseconds();
  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > data = m_Tracker->GetTrackingData();

  m_BackEnd->AddData(this->GetRecordingDirectory(),
                     this->GetIsRecording(),
                     this->GetTimeStampTolerance(),
                     timeCreated,
                     data);

  if (this->GetIsRecording())
  {
    this->SetStatus("Saving");
  }
  else
  {
    this->SetStatus("Grabbing");
  }
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> MITKTrackerDataSourceService::Update(const niftk::IGIDataSourceI::IGITimeType& time)
{
  if (this->GetIsPlayingBack())
  {
    this->PlaybackData(time);
  }

  std::vector<IGIDataItemInfo> infos = m_BackEnd->Update(time);
  return infos;
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  m_BackEnd->SetProperties(properties);
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties MITKTrackerDataSourceService::GetProperties() const
{
  return m_BackEnd->GetProperties();
}

} // end namespace
