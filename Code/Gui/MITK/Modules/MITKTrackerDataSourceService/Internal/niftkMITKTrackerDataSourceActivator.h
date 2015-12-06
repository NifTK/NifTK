/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMITKTrackerDataSourceActivator_h
#define niftkMITKTrackerDataSourceActivator_h

#include "niftkMITKAuroraCubeDataSourceFactory.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* \class MITKTrackerDataSourceActivator
* \brief Activator to register all the MITK Tracker Factories.
*/
class MITKTrackerDataSourceActivator : public us::ModuleActivator
{
public:

  MITKTrackerDataSourceActivator();
  ~MITKTrackerDataSourceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  MITKTrackerDataSourceActivator(const MITKTrackerDataSourceActivator&); // deliberately not implemented
  MITKTrackerDataSourceActivator& operator=(const MITKTrackerDataSourceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::MITKAuroraCubeDataSourceFactory> m_AuroraCubeFactory;
};

} // end namespace

#endif
