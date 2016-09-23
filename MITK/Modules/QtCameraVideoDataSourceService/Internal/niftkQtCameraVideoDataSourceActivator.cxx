/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtCameraVideoDataSourceActivator.h"
#include "niftkQtCameraVideoDataSourceFactory.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
QtCameraVideoDataSourceActivator::QtCameraVideoDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
QtCameraVideoDataSourceActivator::~QtCameraVideoDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataSourceActivator::Load(us::ModuleContext* context)
{
  m_Factory.reset(new QtCameraVideoDataSourceFactory);
  us::ServiceProperties props;
  props["Name"] = std::string("QtCameraVideoDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_Factory.get(), props);
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataSourceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::QtCameraVideoDataSourceActivator)
