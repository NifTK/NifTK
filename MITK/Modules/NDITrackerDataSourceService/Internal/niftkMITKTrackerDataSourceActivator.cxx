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
  us::ServiceProperties auroraCubeProps;
  auroraCubeProps["Name"] = std::string("MITKAuroraCubeDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_AuroraCubeFactory.get(), auroraCubeProps);

  m_AuroraDomeFactory.reset(new MITKAuroraDomeDataSourceFactory);
  us::ServiceProperties auroraDomeProps;
  auroraDomeProps["Name"] = std::string("MITKAuroraDomeDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_AuroraDomeFactory.get(), auroraDomeProps);

  m_AuroraTableTopFactory.reset(new MITKAuroraTableTopDataSourceFactory);
  us::ServiceProperties auroraTableTopProps;
  auroraTableTopProps["Name"] = std::string("MITKAuroraTableTopDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_AuroraTableTopFactory.get(), auroraTableTopProps);

  m_PolarisVicraFactory.reset(new MITKPolarisVicraDataSourceFactory);
  us::ServiceProperties polarisVicraProps;
  polarisVicraProps["Name"] = std::string("MITKPolarisVicraDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_PolarisVicraFactory.get(), polarisVicraProps);

  m_PolarisSpectraFactory.reset(new MITKPolarisSpectraDataSourceFactory);
  us::ServiceProperties polarisSpectraProps;
  polarisSpectraProps["Name"] = std::string("MITKPolarisSpectraDataSourceFactory");
  context->RegisterService<IGIDataSourceFactoryServiceI>(m_PolarisSpectraFactory.get(), polarisSpectraProps);
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceActivator::Unload(us::ModuleContext*)
{
  // NOTE: The services are automatically unregistered
}

} // end namespace

US_EXPORT_MODULE_ACTIVATOR(niftk::MITKTrackerDataSourceActivator)
