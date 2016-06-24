/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOIGTLSystemTimeServiceActivator.h"
#include "niftkOIGTLSystemTimeService.h"
#include <niftkSystemTimeServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
OIGTLSystemTimeServiceActivator::OIGTLSystemTimeServiceActivator()
{
}


//-----------------------------------------------------------------------------
OIGTLSystemTimeServiceActivator::~OIGTLSystemTimeServiceActivator()
{
}


//-----------------------------------------------------------------------------
void OIGTLSystemTimeServiceActivator::Load(us::ModuleContext* context)
{
  m_OIGTLSystemTimeService.reset(new OIGTLSystemTimeService);
  us::ServiceProperties props;
  props["Method"] = std::string("OIGTL");
  context->RegisterService<niftk::SystemTimeServiceI>(m_OIGTLSystemTimeService.get(), props);
}


//-----------------------------------------------------------------------------
void OIGTLSystemTimeServiceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::OIGTLSystemTimeServiceActivator)
