/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNDITracker.h"

#include <mitkException.h>
#include <mitkNavigationToolStorageDeserializer.h>
#include <mitkNavigationToolStorageSerializer.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NDITracker::NDITracker(mitk::DataStorage::Pointer dataStorage,
                       std::string portName,
                       mitk::TrackingDeviceData deviceData,
                       std::string toolConfigFileName,
                       int preferredFramesPerSecond
                       )
: m_DataStorage(dataStorage)
, m_PortName(portName)
, m_DeviceData(deviceData)
, m_ToolConfigFileName(toolConfigFileName)
, m_PreferredFramesPerSecond(preferredFramesPerSecond)
, m_NavigationToolStorage(NULL)
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

  // For polaris at least, the expected number of frames per second depends on number of tools.
  // For Aurora, Im not sure. But this is only used for the late/not-late indicator really.
  m_PreferredFramesPerSecond = m_PreferredFramesPerSecond / m_NavigationToolStorage->GetToolCount();
  MITK_INFO << "Setting tracker to expect " << m_PreferredFramesPerSecond << " frames per second.";

  // Make sure we DONT display surfaces that MITK uses. We just want tracker matrix.
  for (int i = 0; i < m_NavigationToolStorage->GetToolCount(); i++)
  {
    mitk::NavigationTool::Pointer tool = m_NavigationToolStorage->GetTool(i);
    if (tool.IsNull())
    {
      mitkThrow() << "Null tool found in NavigationToolStorage, i=" << i;
    }
    mitk::DataNode::Pointer node = tool->GetDataNode();
    if (node.IsNotNull() && m_DataStorage->Exists(node))
    {
      m_DataStorage->Remove(node);
    }
  }

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
  m_DataStorage->Remove(m_TrackingVolumeNode);
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

} // end namespace
