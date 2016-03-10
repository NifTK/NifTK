/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyLinkDataSourceActivator.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NiftyLinkDataSourceActivator::NiftyLinkDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
NiftyLinkDataSourceActivator::~NiftyLinkDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceActivator::Load(us::ModuleContext* context)
{
  m_NiftyLinkClientFactory.reset(new NiftyLinkClientDataSourceFactory);
  us::ServiceProperties clientProps;
  clientProps["Name"] = std::string("NiftyLinkClientDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_NiftyLinkClientFactory.get(), clientProps);

  m_NiftyLinkServerFactory.reset(new NiftyLinkServerDataSourceFactory);
  us::ServiceProperties serverProps;
  serverProps["Name"] = std::string("NiftyLinkServerDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_NiftyLinkServerFactory.get(), serverProps);
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::NiftyLinkDataSourceActivator)
