/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUltrasonixDataSourceActivator.h"
#include "niftkUltrasonixDataSourceFactory.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
UltrasonixDataSourceActivator::UltrasonixDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceActivator::~UltrasonixDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceActivator::Load(us::ModuleContext* context)
{
  m_Factory.reset(new UltrasonixDataSourceFactory);
  us::ServiceProperties props;
  props["Name"] = std::string("UltrasonixDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_Factory.get(), props);
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::UltrasonixDataSourceActivator)
