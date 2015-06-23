/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkICPRegServiceActivator_h
#define niftkICPRegServiceActivator_h

#include "niftkICPRegService.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* @class ICPRegServiceActivator
* @brief Activator to register instances of niftk::SurfaceRegServiceI, currently only niftk::ICPRegService.
*/
class ICPRegServiceActivator : public us::ModuleActivator
{
public:

  ICPRegServiceActivator();
  ~ICPRegServiceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  ICPRegServiceActivator(const ICPRegServiceActivator&); // deliberately not implemented
  ICPRegServiceActivator& operator=(const ICPRegServiceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::ICPRegService> m_ICPRegService;
};

} // end namespace

#endif
