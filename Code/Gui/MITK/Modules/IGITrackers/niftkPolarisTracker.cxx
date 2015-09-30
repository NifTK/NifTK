/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPolarisTracker.h"

#include <mitkNavigationToolStorageDeserializer.h>
#include <mitkNavigationToolStorageSerializer.h>
#include <mitkException.h>

namespace niftk
{

//-----------------------------------------------------------------------------
PolarisTracker::PolarisTracker(mitk::DataStorage::Pointer dataStorage,
                               mitk::SerialCommunication::PortNumber portNumber,
                               std::string toolConfigFileName)
: m_PreferredFramesPerSecond(60)
, m_DataStorage(dataStorage)
, m_PortNumber(portNumber)
, m_ToolConfigFileName(toolConfigFileName)
, m_NavigationToolStorage(NULL)
, m_TrackingVolumeGenerator(NULL)
, m_TrackingVolumeNode(NULL)
{
  if (dataStorage.IsNull())
  {
    mitkThrow() << "DataStorage is NULL";
  }
  if (toolConfigFileName.size() == 0)
  {
    mitkThrow() << "Empty file name for tracker tool configuration";
  }

  // Load configuration for tracker tools (e.g. pointer, laparoscope etc) from external file.
  mitk::NavigationToolStorageDeserializer::Pointer deserializer = mitk::NavigationToolStorageDeserializer::New(dataStorage);
  m_NavigationToolStorage = deserializer->Deserialize(toolConfigFileName);

  if(m_NavigationToolStorage->isEmpty())
  {
    std::string errorMessage = std::string("Failed to load tracker tool configuration:") + deserializer->GetErrorMessage();
    mitkThrow() << errorMessage;
  }
  if (m_NavigationToolStorage->GetToolCount() < 1)
  {
    mitkThrow() << "No tracker tools available";
  }

  // Setup tracker.
  m_TrackerDevice = mitk::NDITrackingDevice::New();
  m_TrackerDevice->SetType(mitk::NDIPolaris);
  m_TrackerDevice->SetDeviceName("Polaris Spectra");
  m_TrackerDevice->SetPortNumber(portNumber);

  // The point of RAII is that the constructor has successfully acquired all
  // resources, so we should try connecting. The way to disconnect it to delete this object.
  this->OpenConnection();

  // Try loading a volume of interest. This is optional, but do it up-front.
  mitk::TrackingDeviceData data = mitk::GetDeviceDataByName(mitk::DeviceDataPolarisSpectra.Model);
  m_TrackingVolumeGenerator = mitk::TrackingVolumeGenerator::New();
  m_TrackingVolumeGenerator->SetTrackingDeviceData(data);
  m_TrackingVolumeGenerator->Update();

  m_TrackingVolumeNode = mitk::DataNode::New();
  m_TrackingVolumeNode->SetName(data.Model);
  m_TrackingVolumeNode->SetBoolProperty("Backface Culling",true);
  m_TrackingVolumeNode->SetBoolProperty("helper object", true);
  this->SetVisibilityOfTrackingVolume(true);

  mitk::Color red;
  red.SetRed(1);
  m_TrackingVolumeNode->SetColor(red);
  m_TrackingVolumeNode->SetOpacity(0.25);
  m_DataStorage->Add(m_TrackingVolumeNode);
}


//-----------------------------------------------------------------------------
PolarisTracker::~PolarisTracker()
{
  try
  {
    // One should not throw exceptions from a destructor.
    this->StopTracking();
    this->CloseConnection();
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "ERROR: Failed while destroying PolarisTracker:" << e;
  }
}


//-----------------------------------------------------------------------------
void PolarisTracker::OpenConnection()
{
  // You should only call this from constructor.
  if (m_TrackerDevice->GetState() == mitk::TrackingDevice::Setup)
  {
    m_TrackerDevice->OpenConnection();
    if (m_TrackerDevice->GetState() != mitk::TrackingDevice::Ready)
    {
      mitkThrow() << "Failed to connect to tracker";
    }
    MITK_INFO << "Opened connection to polaris on port " << m_PortNumber;
  }
  else
  {
    mitkThrow() << "Tracking device is not setup correctly";
  }
}


//-----------------------------------------------------------------------------
void PolarisTracker::CloseConnection()
{
  // You should only call this from destructor.
  if (m_TrackerDevice->GetState() == mitk::TrackingDevice::Ready)
  {
    m_TrackerDevice->CloseConnection();
    if (m_TrackerDevice->GetState() != mitk::TrackingDevice::Setup)
    {
      mitkThrow() << "Failed to disconnect from tracker";
    }
    MITK_INFO << "Closed connection to polaris on port " << m_PortNumber;
  }
  else
  {
    mitkThrow() << "Tracking device is not setup correctly";
  }
}


//-----------------------------------------------------------------------------
void PolarisTracker::StartTracking()
{
  if (m_TrackerDevice->GetState() == mitk::TrackingDevice::Tracking)
  {
    return;
  }

  m_TrackerDevice->StartTracking();

  if (m_TrackerDevice->GetState() != mitk::TrackingDevice::Tracking)
  {
    mitkThrow() << "Failed to start tracking";
  }
  MITK_INFO << "Started polaris tracking for " << m_TrackerDevice->GetToolCount() << " tools.";
}


//-----------------------------------------------------------------------------
void PolarisTracker::StopTracking()
{
  if (m_TrackerDevice->GetState() == mitk::TrackingDevice::Ready)
  {
    return;
  }

  m_TrackerDevice->StopTracking();

  if (m_TrackerDevice->GetState() != mitk::TrackingDevice::Ready)
  {
    mitkThrow() << "Failed to stop tracking";
  }
  MITK_INFO << "Stopped polaris tracking for " << m_TrackerDevice->GetToolCount() << " tools.";
}


//-----------------------------------------------------------------------------
void PolarisTracker::SetVisibilityOfTrackingVolume(bool isVisible)
{
  m_TrackingVolumeNode->SetBoolProperty("visible", isVisible);
}

} // end namespace
