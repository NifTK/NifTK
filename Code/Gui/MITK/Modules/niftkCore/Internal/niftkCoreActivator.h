/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCoreActivator_h
#define niftkCoreActivator_h

#include "niftkCoordinateAxesDataReaderService.h"
#include "niftkCoordinateAxesDataWriterService.h"
#include <mitkIFileReader.h>
#include <mitkIFileWriter.h>
#include <mitkPNMReader.h>
#include <mitkPNMWriter.h>
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk {

/*
 * This is the module activator for the "niftkCore" module. It registers core services
 */
class CoreActivator : public us::ModuleActivator
{
public:

  CoreActivator();
  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext* );

private:

  std::auto_ptr<CoordinateAxesDataReaderService> m_CoordinateAxesDataReaderService;
  std::auto_ptr<CoordinateAxesDataWriterService> m_CoordinateAxesDataWriterService;
  
  std::auto_ptr<mitk::PNMReader> m_PNMReaderService;
  std::auto_ptr<mitk::PNMWriter> m_PNMWriterService;
};

} // end namespace

#endif
