/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAuroraTableTopTracker.h"

namespace niftk
{

//-----------------------------------------------------------------------------
AuroraTableTopTracker::AuroraTableTopTracker(mitk::DataStorage::Pointer dataStorage,
                                             std::string portName,
                                             std::string toolConfigFileName)
: MITKNDITracker(dataStorage, portName, mitk::DeviceDataAuroraTabletop, toolConfigFileName, 40)
{
}


//-----------------------------------------------------------------------------
AuroraTableTopTracker::~AuroraTableTopTracker()
{
}

} // end namespace
