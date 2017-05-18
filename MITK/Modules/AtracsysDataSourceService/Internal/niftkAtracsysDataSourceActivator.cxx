/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAtracsysDataSourceActivator.h"
#include "niftkAtracsysDataSourceFactory.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
AtracsysDataSourceActivator::AtracsysDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
AtracsysDataSourceActivator::~AtracsysDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
void AtracsysDataSourceActivator::Load(us::ModuleContext* context)
{
  m_Factory.reset(new AtracsysDataSourceFactory);
  us::ServiceProperties props;
  props["Name"] = std::string("AtracsysDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_Factory.get(), props);
}


//-----------------------------------------------------------------------------
void AtracsysDataSourceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::AtracsysDataSourceActivator)
