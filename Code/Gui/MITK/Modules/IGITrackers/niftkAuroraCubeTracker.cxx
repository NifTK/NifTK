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
                                     mitk::SerialCommunication::PortNumber portNumber,
                                     std::string toolConfigFileName)
: NDITracker(dataStorage, portNumber, mitk::NDIAurora, mitk::DeviceDataAuroraPlanarCube, toolConfigFileName, 40)
{
}


//-----------------------------------------------------------------------------
AuroraCubeTracker::~AuroraCubeTracker()
{
  // Smart pointer deletes tracker.
}

} // end namespace
