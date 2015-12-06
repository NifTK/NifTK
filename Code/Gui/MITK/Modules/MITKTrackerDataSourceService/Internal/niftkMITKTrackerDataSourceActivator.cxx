/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKTrackerDataSourceActivator.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MITKTrackerDataSourceActivator::MITKTrackerDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
MITKTrackerDataSourceActivator::~MITKTrackerDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceActivator::Load(us::ModuleContext* context)
{
  m_AuroraCubeFactory.reset(new MITKAuroraCubeDataSourceFactory);
  us::ServiceProperties props;
  props["Name"] = std::string("MITKAuroraCubeDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_AuroraCubeFactory.get(), props);
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::MITKTrackerDataSourceActivator)
