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
                                     mitk::SerialCommunication::PortNumber portNumber,
                                     std::string toolConfigFileName)
{
  this->m_PreferredFramesPerSecond = 40;
  this->m_Tracker = niftk::NDITracker::New(dataStorage, portNumber, mitk::NDIAurora, mitk::DeviceDataAuroraPlanarDome, toolConfigFileName);
}


//-----------------------------------------------------------------------------
AuroraDomeTracker::~AuroraDomeTracker()
{
  // Smart pointer deletes tracker.
}

} // end namespace
