/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSource.h"
#include <niftkSystemTimeServiceRAII.h>

#include <mitkUIDGenerator.h>
#include <mitkExceptionMacro.h>
#include <usModuleContext.h>
#include <usGetModuleContext.h>
#include <cassert>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSource::IGIDataSource(const std::string& name,
                             const std::string& factoryName,
                             mitk::DataStorage::Pointer dataStorage)
: m_Mutex(QMutex::Recursive)
, m_TimeStamp(nullptr)
, m_DataStorage(dataStorage)
, m_Name(QString::fromStdString(name))
, m_FactoryName(QString::fromStdString(factoryName))
, m_Status("UNKNOWN")
, m_Description("UNKNOWN")
, m_RecordingLocation("")
, m_PlaybackSourceName("")
, m_TimeStampTolerance(0)
, m_ShouldUpdate(false)
, m_IsRecording(false)
, m_IsPlayingBack(false)
{
  if (m_DataStorage.IsNull())
  {
    mitkThrow() << "mitk::DataStorage is NULL!";
  }

  if (m_Name.size() == 0)
  {
    mitkThrow() << "Device name is empty!";
  }

  if (m_FactoryName.size() == 0)
  {
    mitkThrow() << "Factory name is empty!";
  }

  m_TimeStamp = igtl::TimeStamp::New();
  m_TimeStamp->GetTime();

  // Register as MicroService.
  mitk::UIDGenerator uidGen = mitk::UIDGenerator ("uk.ac.ucl.cmic.IGIDataSource.id_", 16);

  std::string interfaceName("uk.ac.ucl.cmic.IGIDataSource");
  std::string keyDeviceName = interfaceName + ".device";
  std::string keyId = interfaceName + ".id";

  us::ServiceProperties props;
  props[ keyId ] = uidGen.GetUID();
  props[ keyDeviceName ] = name;

  us::ModuleContext* context = us::GetModuleContext();
  m_MicroServiceRegistration = context->RegisterService(this, props);

  this->Modified();
}


//-----------------------------------------------------------------------------
IGIDataSource::~IGIDataSource()
{
  if(m_MicroServiceRegistration != NULL)
  {
    m_MicroServiceRegistration.Unregister();
  }
  m_MicroServiceRegistration = 0;

  if (m_DataStorage.IsNotNull())
  {
    std::set<mitk::DataNode::Pointer>::iterator iter;
    for (iter = m_DataNodes.begin(); iter != m_DataNodes.end(); ++iter)
    {
      m_DataStorage->Remove(*iter);
    }
  }
}


//-----------------------------------------------------------------------------
QString IGIDataSource::GetName() const
{
  QMutexLocker locker(&m_Mutex);

  return m_Name;
}


//-----------------------------------------------------------------------------
QString IGIDataSource::GetFactoryName() const
{
  QMutexLocker locker(&m_Mutex);

  return m_FactoryName;
}


//-----------------------------------------------------------------------------
QString IGIDataSource::GetStatus() const
{
  QMutexLocker locker(&m_Mutex);

  return m_Status;
}


//-----------------------------------------------------------------------------
void IGIDataSource::SetStatus(const QString& status)
{
  QMutexLocker locker(&m_Mutex);

  m_Status = status;
  this->Modified();

}


//-----------------------------------------------------------------------------
QString IGIDataSource::GetDescription() const
{
  QMutexLocker locker(&m_Mutex);

  return m_Description;
}


//-----------------------------------------------------------------------------
void IGIDataSource::SetDescription(const QString& description)
{
  QMutexLocker locker(&m_Mutex);

  m_Description = description;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer IGIDataSource::GetDataStorage() const
{
  QMutexLocker locker(&m_Mutex);

  return m_DataStorage;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::GetShouldUpdate() const
{
  QMutexLocker locker(&m_Mutex);

  return m_ShouldUpdate;
}


//-----------------------------------------------------------------------------
void IGIDataSource::SetShouldUpdate(bool shouldUpdate)
{
  QMutexLocker locker(&m_Mutex);

  m_ShouldUpdate = shouldUpdate;
  this->Modified();

}


//-----------------------------------------------------------------------------
QString IGIDataSource::GetRecordingLocation() const
{
  QMutexLocker locker(&m_Mutex);

  return m_RecordingLocation;
}


//-----------------------------------------------------------------------------
void IGIDataSource::SetRecordingLocation(const QString& pathName)
{
  QMutexLocker locker(&m_Mutex);

  m_RecordingLocation = pathName;
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSource::SetPlaybackSourceName(const QString& sourceName)
{
  QMutexLocker locker(&m_Mutex);

  m_PlaybackSourceName = sourceName;
  this->Modified();
}


//-----------------------------------------------------------------------------
QString IGIDataSource::GetPlaybackSourceName() const
{
  QMutexLocker locker(&m_Mutex);

  return m_PlaybackSourceName;
}


//-----------------------------------------------------------------------------
QString IGIDataSource::GetPlaybackDirectory() const
{
  QMutexLocker locker(&m_Mutex);

  return this->GetRecordingLocation()
         + QDir::separator()
         + this->GetPlaybackSourceName();
}


//-----------------------------------------------------------------------------
QString IGIDataSource::GetRecordingDirectory() const
{
  QMutexLocker locker(&m_Mutex);

  return this->GetRecordingLocation()
      + QDir::separator()
      + this->GetName();
}


//-----------------------------------------------------------------------------
void IGIDataSource::StartRecording()
{
  QMutexLocker locker(&m_Mutex);

  this->SetIsRecording(true);
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSource::StopRecording()
{
  QMutexLocker locker(&m_Mutex);

  this->SetIsRecording(false);
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSource::StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                                  niftk::IGIDataSourceI::IGITimeType lastTimeStamp)
{
  QMutexLocker locker(&m_Mutex);

  this->SetIsPlayingBack(true);
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSource::StopPlayback()
{
  QMutexLocker locker(&m_Mutex);

  this->SetIsPlayingBack(false);
  this->Modified();
}


//-----------------------------------------------------------------------------
niftk::IGIDataSourceI::IGITimeType IGIDataSource::GetTimeStampInNanoseconds()
{
  QMutexLocker locker(&m_Mutex);

  m_TimeStamp->GetTime();
  return m_TimeStamp->GetTimeStampInNanoseconds();
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer IGIDataSource::GetDataNode(const QString& name, const bool& addToDataStorage)
{
  QMutexLocker locker(&m_Mutex);

  if (m_DataStorage.IsNull())
  {
    mitkThrow() << "DataStorage is NULL!";
  }

  // If name is not specified, use the data source name itself.
  std::string nodeName = name.toStdString();
  if (nodeName.size() == 0)
  {
    nodeName = this->GetName().toStdString();
  }

  // Try and get existing node.
  mitk::DataNode::Pointer result = m_DataStorage->GetNamedNode(nodeName.c_str());

  // If that fails, make one with the right properties.
  if (result.IsNull())
  {
    result = mitk::DataNode::New();
    result->SetVisibility(true);
    result->SetOpacity(1);
    result->SetName(nodeName);

    if (addToDataStorage)
    {
      m_DataStorage->Add(result);
    }
    m_DataNodes.insert(result);
  }

  return result;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::IsLate(const niftk::IGIDataSourceI::IGITimeType& requested,
                           const niftk::IGIDataSourceI::IGITimeType& actual
                          ) const
{
  QMutexLocker locker(&m_Mutex);

  if (actual > requested)
  {
    mitkThrow() << "Retrieved data has a timestamp that is ahead of the requested one, which should never happen";
  }
  return ((requested - actual) > this->GetTimeStampTolerance());
}


//-----------------------------------------------------------------------------
unsigned int IGIDataSource::GetLagInMilliseconds(const niftk::IGIDataSourceI::IGITimeType& requested,
                                                 const niftk::IGIDataSourceI::IGITimeType& actual
                                                ) const
{
  QMutexLocker locker(&m_Mutex);

  if (actual > requested)
  {
    mitkThrow() << "Retrieved data has a timestamp that is ahead of the requested one, which should never happen";
  }
  return (requested - actual)/1000000;
}

} // end namespace
