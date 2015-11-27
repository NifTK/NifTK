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
IGIDataSourceFactoryServiceI::IGIDataSourceFactoryServiceI(std::string displayName,
                                                           std::string service,
                                                           std::string gui,
                                                           bool needGuiAtStartup
                                                           )
: m_DisplayName(displayName)
, m_NameOfService(service)
, m_NameOfGui(gui)
, m_NeedGuiAtStartup(needGuiAtStartup)
{
}


//-----------------------------------------------------------------------------
IGIDataSourceFactoryServiceI::~IGIDataSourceFactoryServiceI()
{
}


//-----------------------------------------------------------------------------
std::string IGIDataSourceFactoryServiceI::GetDisplayName() const
{
  return m_DisplayName;
}


//-----------------------------------------------------------------------------
std::string IGIDataSourceFactoryServiceI::GetNameOfService() const
{
  return m_NameOfService;
}


//-----------------------------------------------------------------------------
std::string IGIDataSourceFactoryServiceI::GetNameOfGui() const
{
  return m_NameOfGui;
}


//-----------------------------------------------------------------------------
bool IGIDataSourceFactoryServiceI::GetNeedGuiAtStartup() const
{
  return m_NeedGuiAtStartup;
}

} // end namespace
