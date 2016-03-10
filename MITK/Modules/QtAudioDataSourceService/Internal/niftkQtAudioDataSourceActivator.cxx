/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtAudioDataSourceActivator.h"
#include "niftkQtAudioDataSourceFactory.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
QtAudioDataSourceActivator::QtAudioDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
QtAudioDataSourceActivator::~QtAudioDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceActivator::Load(us::ModuleContext* context)
{
  m_Factory.reset(new QtAudioDataSourceFactory);
  us::ServiceProperties props;
  props["Name"] = std::string("QtAudioDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_Factory.get(), props);
}


//-----------------------------------------------------------------------------
void QtAudioDataSourceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::QtAudioDataSourceActivator)
