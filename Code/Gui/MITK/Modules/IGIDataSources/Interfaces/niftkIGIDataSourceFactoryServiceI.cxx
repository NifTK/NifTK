/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceFactoryServiceI.h"

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceFactoryServiceI::IGIDataSourceFactoryServiceI(std::string name,
                                                           std::string service,
                                                           bool needGuiAtStartup,
                                                           std::string startupGui,
                                                           std::string observationGui
                                                           )
: m_Name(name)
, m_NameOfService(service)
, m_NeedGuiAtStartup(needGuiAtStartup)
, m_NameOfStartupGui(startupGui)
, m_NameOfObservationGui(observationGui)
{
}


//-----------------------------------------------------------------------------
IGIDataSourceFactoryServiceI::~IGIDataSourceFactoryServiceI()
{
}


//-----------------------------------------------------------------------------
std::string IGIDataSourceFactoryServiceI::GetName() const
{
  return m_Name;
}


//-----------------------------------------------------------------------------
std::string IGIDataSourceFactoryServiceI::GetNameOfService() const
{
  return m_NameOfService;
}


//-----------------------------------------------------------------------------
std::string IGIDataSourceFactoryServiceI::GetNameOfStartupGui() const
{
  return m_NameOfStartupGui;
}


//-----------------------------------------------------------------------------
bool IGIDataSourceFactoryServiceI::GetNeedGuiAtStartup() const
{
  return m_NeedGuiAtStartup;
}


//-----------------------------------------------------------------------------
std::string IGIDataSourceFactoryServiceI::GetNameOfObservationGui() const
{
  return m_NameOfObservationGui;
}

} // end namespace
