/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVicraTracker.h"

namespace niftk
{

//-----------------------------------------------------------------------------
VicraTracker::VicraTracker(mitk::DataStorage::Pointer dataStorage,
                           std::string portName,
                           std::string toolConfigFileName,
                           int baudRate
                           )
: PLUSNDITracker(dataStorage, portName, mitk::DeviceDataPolarisVicra, toolConfigFileName, 20, baudRate, 0)
{
}


//-----------------------------------------------------------------------------
VicraTracker::~VicraTracker()
{
}

} // end namespace
