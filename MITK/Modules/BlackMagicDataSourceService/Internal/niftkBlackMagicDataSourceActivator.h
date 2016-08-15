/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBlackMagicDataSourceActivator_h
#define niftkBlackMagicDataSourceActivator_h

#include "niftkBlackMagicDataSourceFactory.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class BlackMagicDataSourceActivator
* \brief Activator to register the BlackMagicDataSourceFactory.
*/
class BlackMagicDataSourceActivator : public us::ModuleActivator
{
public:

  BlackMagicDataSourceActivator();
  ~BlackMagicDataSourceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  BlackMagicDataSourceActivator(const BlackMagicDataSourceActivator&); // deliberately not implemented
  BlackMagicDataSourceActivator& operator=(const BlackMagicDataSourceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::BlackMagicDataSourceFactory> m_Factory;
};

} // end namespace

#endif
