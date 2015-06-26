/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkICPRegServiceRAII.h"

#include <mitkExceptionMacro.h>
#include <usGetModuleContext.h>
#include "niftkServiceConfigurationI.h"

namespace niftk
{

//-----------------------------------------------------------------------------
ICPRegServiceRAII::ICPRegServiceRAII(const int& maxLandmarks, const int& maxIterations)
: m_ModuleContext(NULL)
, m_Service(NULL)
{
  m_ModuleContext = us::GetModuleContext();

  if (m_ModuleContext == NULL)
  {
    mitkThrow() << "Unable to get us::ModuleContext.";
  }

  m_Refs = m_ModuleContext->GetServiceReferences<SurfaceRegServiceI>("(Method=ICP)");

  if (m_Refs.size() == 0)
  {
    mitkThrow() << "Unable to get us::ServiceReference based on ICP.";
  }

  if (m_Refs.size() > 1)
  {
    MITK_WARN << "Multiple ICP based services are found! This may be a problem." << std::endl;
  }

  m_Service = m_ModuleContext->GetService<niftk::SurfaceRegServiceI>(m_Refs.front());

  if (m_Service == NULL)
  {
    mitkThrow() << "Unable to get niftk::SurfaceRegServiceI.";
  }

  niftk::ServiceConfigurationI *configurableService = dynamic_cast<ServiceConfigurationI*>(m_Service);

  if (configurableService == NULL)
  {
    mitkThrow() << "Retrieved niftk::SurfaceRegServiceI but it was not also a niftk::ServiceConfigurationI";
  }

  us::ServiceProperties props;
  props["MaxLandmarks"] = maxLandmarks;
  props["MaxIterations"] = maxIterations;

  configurableService->Configure(props);
}


//-----------------------------------------------------------------------------
ICPRegServiceRAII::~ICPRegServiceRAII()
{
  m_ModuleContext->UngetService(m_Refs.front());
}


//-----------------------------------------------------------------------------
double ICPRegServiceRAII::Register(
  const mitk::DataNode::Pointer fixedDataSet,
  const mitk::DataNode::Pointer movingDataSet,
  vtkMatrix4x4& matrix) const
{
  return m_Service->Register(fixedDataSet, movingDataSet, matrix);
}

} // end namespace
