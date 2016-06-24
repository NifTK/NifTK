/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkICPRegServiceActivator.h"
#include "niftkICPRegService.h"
#include <niftkSurfaceRegServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
ICPRegServiceActivator::ICPRegServiceActivator()
{
}


//-----------------------------------------------------------------------------
ICPRegServiceActivator::~ICPRegServiceActivator()
{
}


//-----------------------------------------------------------------------------
void ICPRegServiceActivator::Load(us::ModuleContext* context)
{
  m_ICPRegService.reset(new ICPRegService);

  // We could use this one activator to create and hold pointers to many
  // services, each providing different ways of doing point based registration.
  // These services could be distinguished by asking for services with given properties.
  us::ServiceProperties props;
  props["Method"] = std::string("ICP");
  context->RegisterService<SurfaceRegServiceI>(m_ICPRegService.get(), props);
}


//-----------------------------------------------------------------------------
void ICPRegServiceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::ICPRegServiceActivator)
