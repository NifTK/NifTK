/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNDITracker.h"

namespace niftk
{

//-----------------------------------------------------------------------------
NDITracker::NDITracker(mitk::DataStorage::Pointer dataStorage,
                       std::string portName,
                       mitk::TrackingDeviceData deviceData,
                       std::string toolConfigFileName,
                       int preferredFramesPerSecond
                       )
: IGITracker(dataStorage, toolConfigFileName, preferredFramesPerSecond)
, m_PortName(portName)
, m_DeviceData(deviceData)
, m_TrackingVolumeGenerator(nullptr)
{
  m_TrackingVolumeGenerator = mitk::TrackingVolumeGenerator::New();
  m_TrackingVolumeGenerator->SetTrackingDeviceData(m_DeviceData);
  m_TrackingVolumeGenerator->Update();

  // Stored in base class
  m_TrackingVolumeNode = mitk::DataNode::New();
  m_TrackingVolumeNode->SetName(m_DeviceData.Model);
  m_TrackingVolumeNode->SetData(m_TrackingVolumeGenerator->GetOutput());
  m_TrackingVolumeNode->SetBoolProperty("Backface Culling",true);
  m_TrackingVolumeNode->SetBoolProperty("helper object", true);

  mitk::Color red;
  red.SetRed(1);
  red.SetBlue(0);
  red.SetGreen(0);

  m_TrackingVolumeNode->SetColor(red);
  m_TrackingVolumeNode->SetOpacity(0.25);
  this->SetVisibilityOfTrackingVolume(true);

  m_DataStorage->Add(m_TrackingVolumeNode);
}


//-----------------------------------------------------------------------------
NDITracker::~NDITracker()
{
  if (m_DataStorage->Exists(m_TrackingVolumeNode))
  {
    m_DataStorage->Remove(m_TrackingVolumeNode);
  }
}

} // end namespace
