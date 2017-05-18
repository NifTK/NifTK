/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkAtracsysDataSourceActivator_h
#define niftkAtracsysDataSourceActivator_h

#include "niftkAtracsysDataSourceFactory.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class AtracsysDataSourceActivator
* \brief Activator to register the AtracsysDataSourceFactory.
*/
class AtracsysDataSourceActivator : public us::ModuleActivator
{
public:

  AtracsysDataSourceActivator();
  ~AtracsysDataSourceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  AtracsysDataSourceActivator(const AtracsysDataSourceActivator&); // deliberately not implemented
  AtracsysDataSourceActivator& operator=(const AtracsysDataSourceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::AtracsysDataSourceFactory> m_Factory;
};

} // end namespace

#endif
