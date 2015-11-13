/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkLookupTableProviderServiceActivator_h
#define QmitkLookupTableProviderServiceActivator_h

#include "QmitkLookupTableProviderServiceImpl_p.h"
#include "mitkLabelMapReader.h"
#include "mitkLabelMapWriter.h"

#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

/**
 * \class QmitkLookupTableProviderServiceActivator
 */
class US_ABI_LOCAL QmitkLookupTableProviderServiceActivator : public us::ModuleActivator
{

public:

  QmitkLookupTableProviderServiceActivator();

  /** \brief Load module context */
  void Load(us::ModuleContext *context);

  /** \brief Unload module context */
  void Unload(us::ModuleContext* );

private:

  std::auto_ptr<QmitkLookupTableProviderService> m_ServiceImpl;
  
  std::auto_ptr<mitk::LabelMapReader> m_LabelMapReaderService;
  std::auto_ptr<mitk::LabelMapWriter> m_LabelMapWriterService;
};

#endif
