/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkOIGTLSystemTimeServiceActivator_h
#define niftkOIGTLSystemTimeServiceActivator_h

#include "niftkOIGTLSystemTimeService.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class OIGTLSystemTimeServiceActivator
* \brief Activator to register instances of niftk::SystemTimeServiceI.
* \see niftk::OIGTLSystemTimeService
*/
class OIGTLSystemTimeServiceActivator : public us::ModuleActivator
{
public:

  OIGTLSystemTimeServiceActivator();
  ~OIGTLSystemTimeServiceActivator();

  void Load(us::ModuleContext* context) override;
  void Unload(us::ModuleContext*) override;

private:

  OIGTLSystemTimeServiceActivator(const OIGTLSystemTimeServiceActivator&); // deliberately not implemented
  OIGTLSystemTimeServiceActivator& operator=(const OIGTLSystemTimeServiceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::OIGTLSystemTimeService> m_OIGTLSystemTimeService;
};

} // end namespace

#endif
