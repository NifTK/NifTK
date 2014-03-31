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

//-----------------------------------------------------------------------------
void QmitkLookupTableProviderServiceActivator::Load(us::ModuleContext *context)
{
  m_ServiceImpl.reset(new QmitkLookupTableProviderServiceImpl);

  us::ServiceProperties props;
  context->RegisterService<QmitkLookupTableProviderService>(m_ServiceImpl.get(), props);
}

//-----------------------------------------------------------------------------
void QmitkLookupTableProviderServiceActivator::Unload(us::ModuleContext *)
{
}

US_EXPORT_MODULE_ACTIVATOR(niftkCoreGui, QmitkLookupTableProviderServiceActivator )
