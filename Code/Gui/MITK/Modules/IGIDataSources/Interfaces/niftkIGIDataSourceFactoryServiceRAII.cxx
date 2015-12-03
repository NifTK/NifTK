/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceFactoryServiceRAII.h"

#include <mitkExceptionMacro.h>
#include <usGetModuleContext.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceFactoryServiceRAII::IGIDataSourceFactoryServiceRAII(const std::string &factoryName)
: m_ModuleContext(NULL)
, m_Service(NULL)
{
  m_ModuleContext = us::GetModuleContext();

  if (m_ModuleContext == NULL)
  {
    mitkThrow() << "Unable to get us::ModuleContext.";
  }

  m_Refs = m_ModuleContext->GetServiceReferences<IGIDataSourceFactoryServiceI>("(Name=" + factoryName +")");

  if (m_Refs.size() == 0)
  {
    mitkThrow() << "Unable to get us::ServiceReference in IGIDataSourceFactoryServiceRAII(" << factoryName << ").";
  }

  if (m_Refs.size() > 1)
  {
    mitkThrow() << "There should only be 1 us::ServiceReference for IGIDataSourceFactoryServiceRAII(" << factoryName << "), but i found " << m_Refs.size();
  }

  m_Service = m_ModuleContext->GetService<niftk::IGIDataSourceFactoryServiceI>(m_Refs.front());

  if (m_Service == NULL)
  {
    mitkThrow() << "Unable to get niftk::IGIDataSourceFactoryServiceI in IGIDataSourceFactoryServiceRAII(" << factoryName << ").";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceFactoryServiceRAII::~IGIDataSourceFactoryServiceRAII()
{
  m_ModuleContext->UngetService(m_Refs.front());
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer IGIDataSourceFactoryServiceRAII::CreateService(mitk::DataStorage::Pointer dataStorage,
                                                                       const QMap<QString, QVariant>& properties)
{
  return m_Service->CreateService(dataStorage, properties);
}

} // end namespace
