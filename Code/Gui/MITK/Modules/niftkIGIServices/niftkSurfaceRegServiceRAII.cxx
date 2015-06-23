/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSurfaceRegServiceRAII.h"

#include <mitkExceptionMacro.h>
#include <usGetModuleContext.h>

namespace niftk
{

//-----------------------------------------------------------------------------
SurfaceRegServiceRAII::SurfaceRegServiceRAII(const std::string &method)
: m_ModuleContext(NULL)
, m_Service(NULL)
{
  m_ModuleContext = us::GetModuleContext();

  if (m_ModuleContext == NULL)
  {
    mitkThrow() << "Unable to get us::ModuleContext from us::GetModuleContext().";
  }

  m_Refs = m_ModuleContext->GetServiceReferences<SurfaceRegServiceI>("(Method=" + method +")");

  if (m_Refs.size() == 0)
  {
    mitkThrow() << "Unable to get us::ServiceReference in SurfaceRegServiceRAII(" << method << ").";
  }

  m_Service = m_ModuleContext->GetService<niftk::SurfaceRegServiceI>(m_Refs.front());

  if (m_Service == NULL)
  {
    mitkThrow() << "Unable to get niftk::SurfaceRegServiceI in SurfaceRegServiceRAII(" << method << ").";
  }
}


//-----------------------------------------------------------------------------
SurfaceRegServiceRAII::~SurfaceRegServiceRAII()
{
  m_ModuleContext->UngetService(m_Refs.front());
}


//-----------------------------------------------------------------------------
double SurfaceRegServiceRAII::Register(
  const mitk::DataNode::Pointer fixedDataSet,
  const mitk::DataNode::Pointer movingDataSet,
  vtkMatrix4x4& matrix) const
{
  return m_Service->Register(fixedDataSet, movingDataSet, matrix);
}

} // end namespace
