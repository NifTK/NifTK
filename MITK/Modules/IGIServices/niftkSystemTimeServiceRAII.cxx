/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSystemTimeServiceRAII.h"

#include <mitkExceptionMacro.h>
#include <usGetModuleContext.h>

namespace niftk
{

//-----------------------------------------------------------------------------
SystemTimeServiceRAII::SystemTimeServiceRAII()
: m_ModuleContext(NULL)
, m_Service(NULL)
{
  m_ModuleContext = us::GetModuleContext();

  if (m_ModuleContext == NULL)
  {
    mitkThrow() << "Unable to get us::ModuleContext.";
  }

  m_Refs = m_ModuleContext->GetServiceReferences<SystemTimeServiceI>();

  if (m_Refs.size() == 0)
  {
    mitkThrow() << "Unable to get us::ServiceReference in SystemTimeServiceRAII()";
  }

  m_Service = m_ModuleContext->GetService<niftk::SystemTimeServiceI>(m_Refs.front());

  if (m_Service == NULL)
  {
    mitkThrow() << "Unable to get niftk::SystemTimeServiceI in SystemTimeServiceRAII()";
  }
}


//-----------------------------------------------------------------------------
SystemTimeServiceRAII::~SystemTimeServiceRAII()
{
  m_ModuleContext->UngetService(m_Refs.front());
}


//-----------------------------------------------------------------------------
SystemTimeServiceI::TimeType SystemTimeServiceRAII::GetSystemTimeInNanoseconds() const
{
  return m_Service->GetSystemTimeInNanoseconds();
}

} // end namespace
