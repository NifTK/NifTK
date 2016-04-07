/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyLinkDataSourceActivator_h
#define niftkNiftyLinkDataSourceActivator_h

#include "niftkNiftyLinkClientDataSourceFactory.h"
#include "niftkNiftyLinkServerDataSourceFactory.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class NiftyLinkDataSourceActivator
* \brief Activator to register all the Nifty Link factories.
* (should just be a client one, and a server one).
*/
class NiftyLinkDataSourceActivator : public us::ModuleActivator
{
public:

  NiftyLinkDataSourceActivator();
  ~NiftyLinkDataSourceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  NiftyLinkDataSourceActivator(const NiftyLinkDataSourceActivator&); // deliberately not implemented
  NiftyLinkDataSourceActivator& operator=(const NiftyLinkDataSourceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::NiftyLinkClientDataSourceFactory> m_NiftyLinkClientFactory;
  std::auto_ptr<niftk::NiftyLinkServerDataSourceFactory> m_NiftyLinkServerFactory;
};

} // end namespace

#endif
