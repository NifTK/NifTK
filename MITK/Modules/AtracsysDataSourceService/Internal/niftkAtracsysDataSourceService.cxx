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

  m_Tracker = niftk::AtracsysTracker::New(dataStorage, fileName.toStdString());
  m_BackEnd = niftk::IGIMatrixPerFileBackend::New(this->GetName(), this->GetDataStorage());
  m_BackEnd->SetExpectedFramesPerSecond(m_Tracker->GetExpectedFramesPerSecond());

  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
AtracsysDataSourceService::~AtracsysDataSourceService()
{
}


//-----------------------------------------------------------------------------
void AtracsysDataSourceService::StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                                              niftk::IGIDataSourceI::IGITimeType lastTimeStamp)
{

}


//-----------------------------------------------------------------------------
void AtracsysDataSourceService::PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp)
{

}


//-----------------------------------------------------------------------------
void AtracsysDataSourceService::StopPlayback()
{

}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> AtracsysDataSourceService::Update(const niftk::IGIDataSourceI::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;
  return infos;
}


//-----------------------------------------------------------------------------
bool AtracsysDataSourceService::ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                                  niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  return true;
}


//-----------------------------------------------------------------------------
void AtracsysDataSourceService::StartRecording()
{

}


//-----------------------------------------------------------------------------
void AtracsysDataSourceService::StopRecording()
{

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
