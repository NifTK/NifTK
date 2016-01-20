/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNDITracker.h"

#include <mitkNavigationToolStorageDeserializer.h>
#include <mitkNavigationToolStorageSerializer.h>
#include <mitkException.h>
#include <mitkRenderingManager.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NDITracker::NDITracker(mitk::DataStorage::Pointer dataStorage,
                       mitk::SerialCommunication::PortNumber portNumber,
                       mitk::TrackingDeviceData deviceData,
                       std::string toolConfigFileName,
                       int preferredFramesPerSecond
                       )
: m_DataStorage(dataStorage)
, m_PortNumber(portNumber)
, m_DeviceData(deviceData)
, m_ToolConfigFileName(toolConfigFileName)
, m_PreferredFramesPerSecond(preferredFramesPerSecond)
, m_NavigationToolStorage(NULL)
, m_TrackerDevice(NULL)
, m_TrackingVolumeGenerator(NULL)
, m_TrackingVolumeNode(NULL)
{
  if (m_DataStorage.IsNull())
  {
    mitkThrow() << "DataStorage is NULL";
  }
  if (m_ToolConfigFileName.size() == 0)
  {
    mitkThrow() << "Empty file name for tracker tool configuration";
  }

  // Load configuration for tracker tools (e.g. pointer, laparoscope etc) from external file.
  mitk::NavigationToolStorageDeserializer::Pointer deserializer = mitk::NavigationToolStorageDeserializer::New(m_DataStorage);
  m_NavigationToolStorage = deserializer->Deserialize(m_ToolConfigFileName);
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
  m_TrackerDevice->SetData(m_DeviceData);
  m_TrackerDevice->SetPortNumber(m_PortNumber);

  MITK_INFO << "Initialising tracker with " << m_TrackerDevice->GetToolCount() << " tools.";

  // Create source
  m_TrackerSource = mitk::TrackingDeviceSource::New();
  m_TrackerSource->SetTrackingDevice(m_TrackerDevice);

  // Try loading a volume of interest. This is optional, but do it up-front.
  m_TrackingVolumeGenerator = mitk::TrackingVolumeGenerator::New();
  m_TrackingVolumeGenerator->SetTrackingDeviceData(m_DeviceData);
  m_TrackingVolumeGenerator->Update();

  m_TrackingVolumeNode = mitk::DataNode::New();
  m_TrackingVolumeNode->SetName(m_DeviceData.Model);
  m_TrackingVolumeNode->SetData(m_TrackingVolumeGenerator->GetOutput());
  m_TrackingVolumeNode->SetBoolProperty("Backface Culling",true);
  m_TrackingVolumeNode->SetBoolProperty("helper object", true);

  mitk::Color red;
  red.SetRed(1);
  m_TrackingVolumeNode->SetColor(red);
  m_TrackingVolumeNode->SetOpacity(0.25);
  this->SetVisibilityOfTrackingVolume(true);

  m_DataStorage->Add(m_TrackingVolumeNode);
  mitk::RenderingManager::GetInstance()->InitializeViews();

  // The point of RAII is that the constructor has successfully acquired all
  // resources, so we should try connecting. The way to disconnect it to delete this object.
  this->OpenConnection();
  this->StartTracking();
}


//-----------------------------------------------------------------------------
NDITracker::~NDITracker()
{
  try
  {
    // One should not throw exceptions from a destructor.
    this->StopTracking();
    this->CloseConnection();
    m_DataStorage->Remove(m_TrackingVolumeNode);
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "ERROR: Failed while destroying NDITracker:" << e;
  }
}


//-----------------------------------------------------------------------------
void NDITracker::OpenConnection()
{
  // You should only call this from constructor.
  if (m_TrackerDevice->GetState() == mitk::TrackingDevice::Setup)
  {
    m_TrackerSource->Connect();
    if (m_TrackerDevice->GetState() != mitk::TrackingDevice::Ready)
    {
      mitkThrow() << "Failed to connect to tracker";
    }
    MITK_INFO << "Opened connection to tracker on port " << m_PortNumber;
  }
  else
  {
    mitkThrow() << "Tracking device is not setup correctly";
  }
}


//-----------------------------------------------------------------------------
void NDITracker::CloseConnection()
{
  // You should only call this from destructor.
  if (m_TrackerDevice->GetState() == mitk::TrackingDevice::Ready)
  {
    m_TrackerSource->Disconnect();
    if (m_TrackerDevice->GetState() != mitk::TrackingDevice::Setup)
    {
      mitkThrow() << "Failed to disconnect from tracker";
    }
    MITK_INFO << "Closed connection to tracker on port " << m_PortNumber;
  }
  else
  {
    mitkThrow() << "Tracking device is not setup correctly";
  }
}


//-----------------------------------------------------------------------------
void NDITracker::StartTracking()
{
  if (m_TrackerDevice->GetState() == mitk::TrackingDevice::Tracking)
  {
    return;
  }

  m_TrackerSource->StartTracking();

  if (m_TrackerDevice->GetState() != mitk::TrackingDevice::Tracking)
  {
    mitkThrow() << "Failed to start tracking";
  }
  MITK_INFO << "Started tracking for " << m_TrackerDevice->GetToolCount() << " tools.";
}


//-----------------------------------------------------------------------------
void NDITracker::StopTracking()
{
  if (m_TrackerDevice->GetState() == mitk::TrackingDevice::Ready)
  {
    return;
  }

  m_TrackerSource->StopTracking();

  if (m_TrackerDevice->GetState() != mitk::TrackingDevice::Ready)
  {
    mitkThrow() << "Failed to stop tracking";
  }
  MITK_INFO << "Stopped tracking for " << m_TrackerDevice->GetToolCount() << " tools.";
}


//-----------------------------------------------------------------------------
void NDITracker::SetVisibilityOfTrackingVolume(bool isVisible)
{
  m_TrackingVolumeNode->SetBoolProperty("visible", isVisible);
}


//-----------------------------------------------------------------------------
bool NDITracker::GetVisibilityOfTrackingVolume() const
{
  bool result = false;
  m_TrackingVolumeNode->GetBoolProperty("visible", result);
  return result;
}


//-----------------------------------------------------------------------------
std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > NDITracker::GetTrackingData()
{
  m_TrackerSource->Update();
  std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > result;

  for(unsigned int i=0; i< m_TrackerDevice->GetToolCount(); i++)
  {
    mitk::NavigationData::Pointer currentTool = m_TrackerSource->GetOutput(i);
    if(currentTool.IsNotNull() && currentTool->IsDataValid())
    {
      std::string name = currentTool->GetName();
      mitk::Matrix3D rotation = currentTool->GetRotationMatrix();
      mitk::Point3D position = currentTool->GetPosition();
      vtkSmartPointer<vtkMatrix4x4> transform = vtkSmartPointer<vtkMatrix4x4>::New();
      transform->Identity();
      for (int r = 0; r < 3; r++)
      {
        for (int c = 0; c < 3; c++)
        {
          transform->SetElement(r, c, rotation[r][c]);
        }
        transform->SetElement(r, 3, position[r]);
      }
      result.insert(std::pair<std::string, vtkSmartPointer<vtkMatrix4x4> >(name, transform));
    }
  }
  return result;
}

} // end namespace
