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
IGIDataSourceFactoryServiceI::IGIDataSourceFactoryServiceI(std::string factory, std::string service, std::string gui)
: m_NameOfFactory(factory)
, m_NameOfService(service)
, m_NameOfGui(gui)
{
}


//-----------------------------------------------------------------------------
IGIDataSourceFactoryServiceI::~IGIDataSourceFactoryServiceI()
{
}


//-----------------------------------------------------------------------------
std::string IGIDataSourceFactoryServiceI::GetNameOfFactory() const
{
  return m_NameOfFactory;
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


} // end namespace
