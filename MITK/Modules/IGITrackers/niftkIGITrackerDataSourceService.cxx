/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGITrackerDataSourceService.h"
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceLocker IGITrackerDataSourceService::s_Lock;


//-----------------------------------------------------------------------------
IGITrackerDataSourceService::IGITrackerDataSourceService(
  QString name,
  QString factoryName,
  mitk::DataStorage::Pointer dataStorage
)
: IGIDataSource((name + QString("-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Tracker(nullptr)
, m_BackEnd(nullptr)
{
  this->SetStatus("Initialising");

  QString deviceName = this->GetName();
  m_TrackerNumber = (deviceName.remove(0, name.length() + 1)).toInt();
}


//-----------------------------------------------------------------------------
IGITrackerDataSourceService::~IGITrackerDataSourceService()
{
  s_Lock.RemoveSource(m_TrackerNumber);
}


//-----------------------------------------------------------------------------
bool IGITrackerDataSourceService::ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                                    niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  return m_BackEnd->ProbeRecordedData(this->GetPlaybackDirectory(),
                                      firstTimeStampInStore,
                                      lastTimeStampInStore);
}


//-----------------------------------------------------------------------------
void IGITrackerDataSourceService::StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                                                niftk::IGIDataSourceI::IGITimeType lastTimeStamp)
{
  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);
  m_BackEnd->StartPlayback(this->GetPlaybackDirectory(), firstTimeStamp, lastTimeStamp);
}


//-----------------------------------------------------------------------------
void IGITrackerDataSourceService::StopPlayback()
{
  IGIDataSource::StopPlayback();
  m_BackEnd->StopPlayback();
}


//-----------------------------------------------------------------------------
void IGITrackerDataSourceService::PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp)
{
  assert(this->GetIsPlayingBack());

  if (m_BackEnd.IsNull())
  {
    mitkThrow() << "Backend is null. This should not happen! It's a programming bug.";
  }

  m_BackEnd->PlaybackData(this->GetPlaybackDirectory(),
                          this->GetTimeStampTolerance(),
                          requestedTimeStamp);

  this->SetStatus("Playing back");
}


//-----------------------------------------------------------------------------
void IGITrackerDataSourceService::GrabData()
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
std::vector<IGIDataItemInfo> IGITrackerDataSourceService::Update(const niftk::IGIDataSourceI::IGITimeType& time)
{
  if (this->GetIsPlayingBack())
  {
    this->PlaybackData(time);
  }

  std::vector<IGIDataItemInfo> infos = m_BackEnd->Update(time);
  return infos;
}


//-----------------------------------------------------------------------------
void IGITrackerDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  m_BackEnd->SetProperties(properties);
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties IGITrackerDataSourceService::GetProperties() const
{
  return m_BackEnd->GetProperties();
}

} // end namespace
