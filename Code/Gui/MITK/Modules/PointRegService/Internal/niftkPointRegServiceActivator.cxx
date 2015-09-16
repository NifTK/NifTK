/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPointRegServiceActivator.h"
#include "niftkPointRegServiceUsingSVD.h"
#include <niftkPointRegServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
PointRegServiceActivator::PointRegServiceActivator()
{
}


//-----------------------------------------------------------------------------
PointRegServiceActivator::~PointRegServiceActivator()
{
}


//-----------------------------------------------------------------------------
void PointRegServiceActivator::Load(us::ModuleContext* context)
{
  m_PointRegServiceSVD.reset(new PointRegServiceUsingSVD);

  // We could use this one activator to create and hold pointers to many
  // services, each providing different ways of doing point based registration.
  // These services could be distinguished by asking for services with given properties.
  us::ServiceProperties props;
  props["Method"] = std::string("SVD");
  context->RegisterService<PointRegServiceI>(m_PointRegServiceSVD.get(), props);
}


//-----------------------------------------------------------------------------
void PointRegServiceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::PointRegServiceActivator)
