/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkQtCameraVideoDataSourceActivator_h
#define niftkQtCameraVideoDataSourceActivator_h

#include "niftkQtCameraVideoDataSourceFactory.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class QtCameraVideoDataSourceActivator
* \brief Activator to register the QtCameraVideoDataSourceFactory.
*/
class QtCameraVideoDataSourceActivator : public us::ModuleActivator
{
public:

  QtCameraVideoDataSourceActivator();
  ~QtCameraVideoDataSourceActivator();

  void Load(us::ModuleContext* context) override;
  void Unload(us::ModuleContext*) override;

private:

  QtCameraVideoDataSourceActivator(const QtCameraVideoDataSourceActivator&); // deliberately not implemented
  QtCameraVideoDataSourceActivator& operator=(const QtCameraVideoDataSourceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::QtCameraVideoDataSourceFactory> m_Factory;
};

} // end namespace

#endif
