/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBlackMagicDataSourceService.h"
#include "niftkBlackMagicDataType.h"
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataSourceUtils.h>
#include <mitkExceptionMacro.h>
#include <QDir>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker BlackMagicDataSourceService::s_Lock;

//-----------------------------------------------------------------------------
BlackMagicDataSourceService::BlackMagicDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("BlackMagic") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
{
  mitkThrow() << "Not implemented yet. Volunteers .... please step forward!";

  this->SetStatus("Initialising");

  QString deviceName = this->GetName();
  m_ChannelNumber = (deviceName.remove(0, 10)).toInt(); // Should match string BlackMagic above

  this->SetShouldUpdate(true);
  this->SetProperties(properties);

  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
BlackMagicDataSourceService::~BlackMagicDataSourceService()
{
  s_Lock.RemoveSource(m_ChannelNumber);
}


//-----------------------------------------------------------------------------
void BlackMagicDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties BlackMagicDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  return props;
}


//-----------------------------------------------------------------------------
void BlackMagicDataSourceService::StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                                                 niftk::IGIDataSourceI::IGITimeType lastTimeStamp)
{
}


//-----------------------------------------------------------------------------
void BlackMagicDataSourceService::StopPlayback()
{
}


//-----------------------------------------------------------------------------
void BlackMagicDataSourceService::PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp)
{
}


//-----------------------------------------------------------------------------
bool BlackMagicDataSourceService::ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                                    niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  niftk::IGIDataSourceI::IGITimeType  firstTimeStampFound = 0;
  niftk::IGIDataSourceI::IGITimeType  lastTimeStampFound  = 0;

  return firstTimeStampFound != 0;
}


//-----------------------------------------------------------------------------
void BlackMagicDataSourceService::GrabData()
{
  {
    QMutexLocker locker(&m_Lock);

    if (this->GetIsPlayingBack())
    {
      return;
    }
  }

  // Save synchronously.
  // This has the side effect that if saving is too slow,
  // the QTimers just won't keep up, and start missing pulses.
  if (this->GetIsRecording())
  {
    //this->SaveItem(wrapper.GetPointer());
  }
  this->SetStatus("Grabbing");
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> BlackMagicDataSourceService::Update(const niftk::IGIDataSourceI::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;

  // Create default return status.
  IGIDataItemInfo info;
  info.m_Name = this->GetName();
  info.m_IsLate = false;
  info.m_FramesPerSecond = 0;
  info.m_LagInMilliseconds = 0;
  infos.push_back(info);
  return infos;
}

} // end namespace
