/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAuroraDomeTracker.h"

namespace niftk
{

//-----------------------------------------------------------------------------
AuroraDomeTracker::AuroraDomeTracker(mitk::DataStorage::Pointer dataStorage,
                                     std::string portName,
                                     std::string toolConfigFileName)
: PLUSNDITracker(dataStorage, portName, mitk::DeviceDataAuroraPlanarDome, toolConfigFileName, 40, 2)
{
}


//-----------------------------------------------------------------------------
AuroraDomeTracker::~AuroraDomeTracker()
{
}

} // end namespace
