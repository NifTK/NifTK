/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNVidiaSDIDataSourceActivator_h
#define niftkNVidiaSDIDataSourceActivator_h

#include "niftkNVidiaSDIDataSourceFactory.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class NVidiaSDIDataSourceActivator
* \brief Activator to register the NVidiaSDIDataSourceFactory.
*/
class NVidiaSDIDataSourceActivator : public us::ModuleActivator
{
public:

  NVidiaSDIDataSourceActivator();
  ~NVidiaSDIDataSourceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  NVidiaSDIDataSourceActivator(const NVidiaSDIDataSourceActivator&); // deliberately not implemented
  NVidiaSDIDataSourceActivator& operator=(const NVidiaSDIDataSourceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::NVidiaSDIDataSourceFactory> m_Factory;
};

} // end namespace

#endif
