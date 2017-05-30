/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAtracsysDataSourceService.h"
#include <niftkAtracsysTracker.h>
#include <niftkIGIMatrixPerFileBackend.h>

#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker AtracsysDataSourceService::s_Lock;


//-----------------------------------------------------------------------------
AtracsysDataSourceService::AtracsysDataSourceService(
  QString factoryName,
  const IGIDataSourceProperties& properties,
  mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("Atracsys-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage
               )
{
  if(!properties.contains("file"))
  {
    mitkThrow() << "Config file name not specified!";
  }
  QString fileName = (properties.value("file")).toString();

  this->SetStatus("Initialising");

  QString deviceName = this->GetName();
  m_TrackerNumber = (deviceName.remove(0, 9)).toInt();

  m_Tracker = niftk::AtracsysTracker::New(dataStorage, fileName.toStdString());
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

  this->SetDescription("Atracsys Tracker:" + this->GetName());

  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
AtracsysDataSourceService::~AtracsysDataSourceService()
{
  m_DataGrabbingThread->ForciblyStop();
  delete m_DataGrabbingThread;

  s_Lock.RemoveSource(m_TrackerNumber);
}


//-----------------------------------------------------------------------------
bool AtracsysDataSourceService::ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                                  niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  return m_BackEnd->ProbeRecordedData(this->GetPlaybackDirectory(),
                                      firstTimeStampInStore,
                                      lastTimeStampInStore);
}


//-----------------------------------------------------------------------------
void AtracsysDataSourceService::StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                                              niftk::IGIDataSourceI::IGITimeType lastTimeStamp)
{
  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);
  m_BackEnd->StartPlayback(this->GetPlaybackDirectory(), firstTimeStamp, lastTimeStamp);
}


//-----------------------------------------------------------------------------
void AtracsysDataSourceService::StopPlayback()
{
  IGIDataSource::StopPlayback();
  m_BackEnd->StopPlayback();
}


//-----------------------------------------------------------------------------
void AtracsysDataSourceService::PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp)
{
  assert(this->GetIsPlayingBack());

  m_BackEnd->PlaybackData(this->GetPlaybackDirectory(),
                          this->GetTimeStampTolerance(),
                          requestedTimeStamp);

  this->SetStatus("Playing back");
}


//-----------------------------------------------------------------------------
void AtracsysDataSourceService::GrabData()
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
std::vector<IGIDataItemInfo> AtracsysDataSourceService::Update(const niftk::IGIDataSourceI::IGITimeType& time)
{
  if (this->GetIsPlayingBack())
  {
    this->PlaybackData(time);
  }

  std::vector<IGIDataItemInfo> infos = m_BackEnd->Update(time);
  return infos;
}


//-----------------------------------------------------------------------------
void AtracsysDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  m_BackEnd->SetProperties(properties);
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties AtracsysDataSourceService::GetProperties() const
{
  return m_BackEnd->GetProperties();
}

} // end namespace
