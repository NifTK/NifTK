/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkLookupTableProviderServiceActivator.h"
#include "QmitkLookupTableProviderServiceImpl_p.h"
#include "niftkCoreGuiIOMimeTypes.h"

//-----------------------------------------------------------------------------
QmitkLookupTableProviderServiceActivator::QmitkLookupTableProviderServiceActivator()
: m_LabelMapReaderService(NULL)
, m_LabelMapWriterService(NULL)
{
}


//-----------------------------------------------------------------------------
void QmitkLookupTableProviderServiceActivator::Load(us::ModuleContext *context)
{
  m_ServiceImpl.reset(new QmitkLookupTableProviderServiceImpl);
  
  us::ServiceProperties props;
  context->RegisterService<QmitkLookupTableProviderService>(m_ServiceImpl.get(), props);
  props[ us::ServiceConstants::SERVICE_RANKING() ] = 10;

  std::vector<mitk::CustomMimeType*> mimeTypes = niftk::CoreGuiIOMimeTypes::Get();
  for (std::vector<mitk::CustomMimeType*>::const_iterator mimeTypeIter = mimeTypes.begin(),
    iterEnd = mimeTypes.end(); mimeTypeIter != iterEnd; ++mimeTypeIter)
  {
    context->RegisterService(*mimeTypeIter, props);
  }

  m_LabelMapReaderService.reset(new mitk::LabelMapReader());
  m_LabelMapWriterService.reset(new mitk::LabelMapWriter());
}


//-----------------------------------------------------------------------------
void QmitkLookupTableProviderServiceActivator::Unload(us::ModuleContext *)
{
  m_LabelMapReaderService.reset(NULL);
  m_LabelMapWriterService.reset(NULL);
}

US_EXPORT_MODULE_ACTIVATOR(QmitkLookupTableProviderServiceActivator)
