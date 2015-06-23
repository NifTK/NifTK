/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPointRegServiceRAII.h"

#include <mitkExceptionMacro.h>
#include <usGetModuleContext.h>

namespace niftk
{

//-----------------------------------------------------------------------------
PointRegServiceRAII::PointRegServiceRAII(const std::string &method)
: m_ModuleContext(NULL)
, m_Service(NULL)
{
  m_ModuleContext = us::GetModuleContext();

  if (m_ModuleContext == NULL)
  {
    mitkThrow() << "Unable to get us::ModuleContext from us::GetModuleContext().";
  }

  m_Refs = m_ModuleContext->GetServiceReferences<PointRegServiceI>("(Method=" + method +")");

  if (m_Refs.size() == 0)
  {
    mitkThrow() << "Unable to get us::ServiceReference in PointRegServiceRAII(" << method << ").";
  }

  m_Service = m_ModuleContext->GetService<niftk::PointRegServiceI>(m_Refs.front());

  if (m_Service == NULL)
  {
    mitkThrow() << "Unable to get niftk::PointRegServiceI in PointRegServiceRAII(" << method << ").";
  }
}


//-----------------------------------------------------------------------------
PointRegServiceRAII::~PointRegServiceRAII()
{
  m_ModuleContext->UngetService(m_Refs.front());
}


//-----------------------------------------------------------------------------
double PointRegServiceRAII::Register(
  const mitk::PointSet::Pointer fixedPoints,
  const mitk::PointSet::Pointer movingPoints,
  vtkMatrix4x4& matrix) const
{
  return m_Service->Register(fixedPoints, movingPoints, matrix);
}

} // end namespace
