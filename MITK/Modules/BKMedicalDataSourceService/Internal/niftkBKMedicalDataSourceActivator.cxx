/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBKMedicalDataSourceActivator.h"
#include "niftkBKMedicalDataSourceFactory.h"
#include <niftkIGIDataSourceFactoryServiceI.h>
#include <usServiceProperties.h>

namespace niftk
{

//-----------------------------------------------------------------------------
BKMedicalDataSourceActivator::BKMedicalDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
BKMedicalDataSourceActivator::~BKMedicalDataSourceActivator()
{
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceActivator::Load(us::ModuleContext* context)
{
  m_Factory.reset(new BKMedicalDataSourceFactory);
  us::ServiceProperties props;
  props["Name"] = std::string("BKMedicalDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_Factory.get(), props);
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::BKMedicalDataSourceActivator)
