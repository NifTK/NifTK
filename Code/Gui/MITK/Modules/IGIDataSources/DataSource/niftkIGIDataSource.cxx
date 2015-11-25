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

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSource::IGIDataSource(const std::string& microServiceDeviceName, mitk::DataStorage::Pointer dataStorage)
: m_DataStorage(dataStorage)
, m_MicroServiceDeviceName(microServiceDeviceName)
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
}

} // end namespace

