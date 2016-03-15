/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPLUSNDITracker.h"

namespace niftk
{

//-----------------------------------------------------------------------------
PLUSNDITracker::PLUSNDITracker(mitk::DataStorage::Pointer dataStorage,
                               std::string portName,
                               mitk::TrackingDeviceData deviceData,
                               std::string toolConfigFileName,
                               int preferredFramesPerSecond
                               )
: NDITracker(dataStorage, portName, deviceData, toolConfigFileName, preferredFramesPerSecond)
{
}


//-----------------------------------------------------------------------------
PLUSNDITracker::~PLUSNDITracker()
{
}

} // end namespace
