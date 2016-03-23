/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAuroraCubeTracker.h"

namespace niftk
{

//-----------------------------------------------------------------------------
AuroraCubeTracker::AuroraCubeTracker(mitk::DataStorage::Pointer dataStorage,
                                     std::string portName,
                                     std::string toolConfigFileName)
: MITKNDITracker(dataStorage, portName, mitk::DeviceDataAuroraPlanarCube, toolConfigFileName, 40)
{
}


//-----------------------------------------------------------------------------
AuroraCubeTracker::~AuroraCubeTracker()
{
}

} // end namespace
