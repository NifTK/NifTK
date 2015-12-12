/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkQtAudioDataSourceActivator_h
#define niftkQtAudioDataSourceActivator_h

#include "niftkQtAudioDataSourceFactory.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class QtAudioDataSourceActivator
* \brief Activator to register the QtAudioDataSourceFactory.
*/
class QtAudioDataSourceActivator : public us::ModuleActivator
{
public:

  QtAudioDataSourceActivator();
  ~QtAudioDataSourceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  QtAudioDataSourceActivator(const QtAudioDataSourceActivator&); // deliberately not implemented
  QtAudioDataSourceActivator& operator=(const QtAudioDataSourceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::QtAudioDataSourceFactory> m_Factory;
};

} // end namespace

#endif
