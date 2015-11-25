/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkOpenCVVideoDataSourceActivator_h
#define niftkOpenCVVideoDataSourceActivator_h

#include "niftkOpenCVVideoDataSourceFactory.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class OpenCVVideoDataSourceActivator
* \brief Activator to register the OpenCVVideoDataSourceFactory.
*/
class OpenCVVideoDataSourceActivator : public us::ModuleActivator
{
public:

  OpenCVVideoDataSourceActivator();
  ~OpenCVVideoDataSourceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  OpenCVVideoDataSourceActivator(const OpenCVVideoDataSourceActivator&); // deliberately not implemented
  OpenCVVideoDataSourceActivator& operator=(const OpenCVVideoDataSourceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::OpenCVVideoDataSourceFactory> m_Factory;
};

} // end namespace

#endif
