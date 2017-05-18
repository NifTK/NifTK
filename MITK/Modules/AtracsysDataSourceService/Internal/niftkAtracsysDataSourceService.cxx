/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAtracsysDataSourceService.h"
#include <mitkExceptionMacro.h>

namespace niftk
{

const int AtracsysDataSourceService::ATRACSYS_FRAMES_PER_SECOND(330);
const int AtracsysDataSourceService::ATRACSYS_TIMEOUT(1000); // ms

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
  // In contrast say, to the OpenCV source, we don't set the lag
  // directly on the buffer because, there may be no buffers present
  // at the time this method is called. For example, you could
  // have created a tracker, and no tracked objects are placed within
  // the field of view, thereby no tracking matrices would have been generated.
  if (properties.contains("lag"))
  {
    int milliseconds = (properties.value("lag")).toInt();
    m_Lag = milliseconds;

    MITK_INFO << "AtracsysDataSourceService(" << this->GetName().toStdString()
              << "): set lag to " << milliseconds << " ms.";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties AtracsysDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  props.insert("lag", m_Lag);

  MITK_INFO << "AtracsysDataSourceService:(" << this->GetName().toStdString()
            << "): Retrieved current value of lag as " << m_Lag << " ms.";

  return props;
}

} // end namespace
