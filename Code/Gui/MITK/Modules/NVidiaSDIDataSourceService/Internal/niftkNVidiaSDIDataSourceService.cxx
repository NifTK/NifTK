/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNVidiaSDIDataSourceService.h"
#include "niftkNVidiaSDIDataType.h"
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataSourceUtils.h>
#include <mitkExceptionMacro.h>
#include <QDir>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker NVidiaSDIDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
NVidiaSDIDataSourceService::NVidiaSDIDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("NVidiaSDI-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_FrameId(0)
{
  this->SetStatus("Initialising");

  QString deviceName = this->GetName();
  m_ChannelNumber = (deviceName.remove(0, 10)).toInt(); // Should match string NVidiaSDI- above

  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
NVidiaSDIDataSourceService::~NVidiaSDIDataSourceService()
{
  s_Lock.RemoveSource(m_ChannelNumber);
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties NVidiaSDIDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  return props;
}


//-----------------------------------------------------------------------------
QString NVidiaSDIDataSourceService::GetRecordingDirectoryName()
{
  return this->GetRecordingLocation()
      + niftk::GetPreferredSlash()
      + this->GetName()
      + "_" + (tr("%1").arg(m_ChannelNumber))
      ;
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                                               niftk::IGIDataType::IGITimeType lastTimeStamp)
{
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceService::StopPlayback()
{
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceService::PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp)
{
}


//-----------------------------------------------------------------------------
bool NVidiaSDIDataSourceService::ProbeRecordedData(const QString& path,
                                                     niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                                                     niftk::IGIDataType::IGITimeType* lastTimeStampInStore)
{
  // zero is a suitable default value. it's unlikely that anyone recorded a legitime data set in the middle ages.
  niftk::IGIDataType::IGITimeType  firstTimeStampFound = 0;
  niftk::IGIDataType::IGITimeType  lastTimeStampFound  = 0;

  return firstTimeStampFound != 0;
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> NVidiaSDIDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;

  IGIDataItemInfo info;
  info.m_Name = this->GetName();
  info.m_FramesPerSecond = 0;
  info.m_IsLate = false;
  info.m_LagInMilliseconds = 0;
  infos.push_back(info);

  return infos;
}

} // end namespace
