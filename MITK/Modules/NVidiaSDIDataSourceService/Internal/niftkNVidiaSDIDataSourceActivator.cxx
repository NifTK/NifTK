/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNVidiaSDIDataSourceActivator.h"
#include "niftkNVidiaSDIDataSourceFactory.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NVidiaSDIDataSourceActivator::NVidiaSDIDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
NVidiaSDIDataSourceActivator::~NVidiaSDIDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceActivator::Load(us::ModuleContext* context)
{
  m_Factory.reset(new NVidiaSDIDataSourceFactory);
  us::ServiceProperties props;
  props["Name"] = std::string("NVidiaSDIDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_Factory.get(), props);
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::NVidiaSDIDataSourceActivator)
