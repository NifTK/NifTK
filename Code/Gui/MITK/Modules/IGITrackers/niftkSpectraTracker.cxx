/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSpectraTracker.h"

namespace niftk
{

//-----------------------------------------------------------------------------
SpectraTracker::SpectraTracker(mitk::DataStorage::Pointer dataStorage,
                               mitk::SerialCommunication::PortNumber portNumber,
                               std::string toolConfigFileName)
{
  this->m_PreferredFramesPerSecond = 60;
  this->m_Tracker = niftk::NDITracker::New(dataStorage, portNumber, mitk::NDIPolaris, mitk::DeviceDataPolarisSpectra, toolConfigFileName);
}


//-----------------------------------------------------------------------------
SpectraTracker::~SpectraTracker()
{
  // Smart pointer deletes tracker.
}

} // end namespace
