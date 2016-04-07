/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUltrasonixDataSourceActivator_h
#define niftkUltrasonixDataSourceActivator_h

#include "niftkUltrasonixDataSourceFactory.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class UltrasonixDataSourceActivator
* \brief Activator to register the UltrasonixDataSourceFactory.
*/
class UltrasonixDataSourceActivator : public us::ModuleActivator
{
public:

  UltrasonixDataSourceActivator();
  ~UltrasonixDataSourceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  UltrasonixDataSourceActivator(const UltrasonixDataSourceActivator&); // deliberately not implemented
  UltrasonixDataSourceActivator& operator=(const UltrasonixDataSourceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::UltrasonixDataSourceFactory> m_Factory;
};

} // end namespace

#endif
