/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSource.h"

#include <mitkUIDGenerator.h>
#include <mitkExceptionMacro.h>
#include <usModuleContext.h>
#include <usGetModuleContext.h>
#include <boost/filesystem.hpp>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSource::IGIDataSource(const std::string& microServiceDeviceName,
                             mitk::DataStorage::Pointer dataStorage)
: m_DataStorage(dataStorage)
, m_TimeCreated(NULL)
, m_MicroServiceDeviceName(microServiceDeviceName)
, m_Status("UNKNOWN")
, m_ShouldUpdate(false)
, m_TimeStampTolerance(0)
{

  if (m_DataStorage.IsNull())
  {
    mitkThrow() << "mitk::DataStorage is NULL!";
  }

  if (m_MicroServiceDeviceName.size() == 0)
  {
    mitkThrow() << "Device name is empty!";
  }

  m_TimeCreated = igtl::TimeStamp::New();
  m_TimeCreated->GetTime();

  // Register as MicroService.
  mitk::UIDGenerator uidGen = mitk::UIDGenerator ("uk.ac.ucl.cmic.IGIDataSource.id_", 16);

  std::string interfaceName("uk.ac.ucl.cmic.IGIDataSource");
  std::string keyDeviceName = interfaceName + ".device";
  std::string keyId = interfaceName + ".id";

  us::ServiceProperties props;
  props[ keyId ] = uidGen.GetUID();
  props[ keyDeviceName ] = microServiceDeviceName;

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
    for (iter = m_DataNodes.begin(); iter != m_DataNodes.end(); iter++)
    {
      m_DataStorage->Remove(*iter);
    }
  }
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer IGIDataSource::GetDataStorage() const
{
  return m_DataStorage;
}


//-----------------------------------------------------------------------------
niftk::IGIDataType::IGITimeType IGIDataSource::GetTimeStampInNanoseconds()
{
  m_TimeCreated->GetTime();
  return m_TimeCreated->GetTimeStampInNanoseconds();
}


//-----------------------------------------------------------------------------
std::string IGIDataSource::GetName() const
{
  return this->GetMicroServiceDeviceName();
}


//-----------------------------------------------------------------------------
std::string IGIDataSource::GetStatus() const
{
  return m_Status;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::GetShouldUpdate() const
{
  return m_ShouldUpdate;
}


//-----------------------------------------------------------------------------
void IGIDataSource::SetShouldUpdate(bool shouldUpdate)
{
  m_ShouldUpdate = shouldUpdate;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer IGIDataSource::GetDataNode(const std::string& name, const bool& addToDataStorage)
{
  if (m_DataStorage.IsNull())
  {
    mitkThrow() << "DataStorage is NULL!";
  }

  // If name is not specified, use the data source name itself.
  std::string nodeName = name;
  if (nodeName.size() == 0)
  {
    nodeName = this->GetMicroServiceDeviceName();
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
std::string IGIDataSource::GetPreferredSlash() const
{
  boost::filesystem::path slash("/");
  std::string preferredSlash = slash.make_preferred().string();
  return preferredSlash;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::IsLate(const niftk::IGIDataType::IGITimeType& requested,
                           const niftk::IGIDataType::IGITimeType& actual
                          ) const
{
  if (actual > requested)
  {
    mitkThrow() << "Retrieved data has a timestamp that is ahead of the requested one, which should never happen";
  }
  return ((requested - actual) > this->GetTimeStampTolerance());
}


//-----------------------------------------------------------------------------
unsigned int IGIDataSource::GetLagInMilliseconds(const niftk::IGIDataType::IGITimeType& requested,
                                                 const niftk::IGIDataType::IGITimeType& actual
                                                ) const
{
  if (actual > requested)
  {
    mitkThrow() << "Retrieved data has a timestamp that is ahead of the requested one, which should never happen";
  }
  return (requested - actual)/1000000;
}

} // end namespace

