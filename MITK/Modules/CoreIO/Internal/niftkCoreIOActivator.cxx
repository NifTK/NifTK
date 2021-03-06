/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCoreIOActivator.h"

#include <usModuleContext.h>

#include "niftkCoreIOMimeTypes.h"
#include "niftkLookupTableProviderServiceImpl_p.h"


namespace niftk
{

//-----------------------------------------------------------------------------
CoreIOActivator::CoreIOActivator()
  : m_CoordinateAxesDataReaderService(NULL),
    m_CoordinateAxesDataWriterService(NULL),
    m_PNMReaderService(NULL),
    m_PNMWriterService(NULL),
    m_LookupTableProviderService(NULL),
    m_LabelMapReaderService(NULL),
    m_LabelMapWriterService(NULL)
{
}


//-----------------------------------------------------------------------------
void CoreIOActivator::Load(us::ModuleContext* context)
{
  m_LookupTableProviderService.reset(new LookupTableProviderServiceImpl);

  us::ServiceProperties props;
  context->RegisterService<LookupTableProviderService>(m_LookupTableProviderService.get(), props);

  props[ us::ServiceConstants::SERVICE_RANKING() ] = 10;

  std::vector<mitk::CustomMimeType*> mimeTypes = niftk::CoreIOMimeTypes::Get();
  for (std::vector<mitk::CustomMimeType*>::const_iterator mimeTypeIter = mimeTypes.begin(),
    iterEnd = mimeTypes.end();
    mimeTypeIter != iterEnd;
    ++mimeTypeIter)
  {
    context->RegisterService(*mimeTypeIter, props);
  }

  m_CoordinateAxesDataReaderService.reset(new niftk::CoordinateAxesDataReaderService());
  m_CoordinateAxesDataWriterService.reset(new niftk::CoordinateAxesDataWriterService());

  m_PNMReaderService.reset(new niftk::PNMReaderService());
  m_PNMWriterService.reset(new niftk::PNMWriterService());

  m_LabelMapReaderService.reset(new niftk::LabelMapReader());
  m_LabelMapWriterService.reset(new niftk::LabelMapWriter());
}


//-----------------------------------------------------------------------------
void CoreIOActivator::Unload(us::ModuleContext*)
{
  m_CoordinateAxesDataReaderService.reset(NULL);
  m_CoordinateAxesDataWriterService.reset(NULL);

  m_PNMReaderService.reset(NULL);
  m_PNMWriterService.reset(NULL);

  m_LookupTableProviderService.reset(NULL);
  m_LabelMapReaderService.reset(NULL);
  m_LabelMapWriterService.reset(NULL);
}

}

US_EXPORT_MODULE_ACTIVATOR(niftk::CoreIOActivator)
