/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBlackMagicDataSourceActivator.h"
#include "niftkBlackMagicDataSourceFactory.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
BlackMagicDataSourceActivator::BlackMagicDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
BlackMagicDataSourceActivator::~BlackMagicDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
void BlackMagicDataSourceActivator::Load(us::ModuleContext* context)
{
  m_Factory.reset(new BlackMagicDataSourceFactory);
  us::ServiceProperties props;
  props["Name"] = std::string("BlackMagicDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_Factory.get(), props);
}


//-----------------------------------------------------------------------------
void BlackMagicDataSourceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::BlackMagicDataSourceActivator)
