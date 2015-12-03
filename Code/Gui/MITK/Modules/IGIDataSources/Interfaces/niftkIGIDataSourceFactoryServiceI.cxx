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
                                                           bool hasInitialiseGui,
                                                           bool hasConfigurationGui
                                                           )
: m_Name(name)
, m_HasInitialiseGui(hasInitialiseGui)
, m_HasConfigurationGui(hasConfigurationGui)
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
bool IGIDataSourceFactoryServiceI::HasInitialiseGui() const
{
  return m_HasInitialiseGui;
}


//-----------------------------------------------------------------------------
bool IGIDataSourceFactoryServiceI::HasConfigurationGui() const
{
  return m_HasConfigurationGui;
}

} // end namespace
