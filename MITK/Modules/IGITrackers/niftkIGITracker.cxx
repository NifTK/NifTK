/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGITracker.h"
#include <niftkMITKMathsUtils.h>

#include <mitkException.h>
#include <mitkNavigationToolStorageDeserializer.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGITracker::IGITracker(mitk::DataStorage::Pointer dataStorage,
                       std::string toolConfigFileName,
                       int expectedFramesPerSecond
                       )
: m_DataStorage(dataStorage)
, m_ToolConfigFileName(toolConfigFileName)
, m_ExpectedFramesPerSecond(expectedFramesPerSecond)
, m_TrackingVolumeNode(nullptr)
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
  mitk::NavigationToolStorageDeserializer::Pointer deserializer
      = mitk::NavigationToolStorageDeserializer::New(m_DataStorage);

  m_NavigationToolStorage = deserializer->Deserialize(m_ToolConfigFileName);
  if(m_NavigationToolStorage->isEmpty())
  {
    std::string errorMessage = std::string("Failed to load tracker tool configuration:")
                               + deserializer->GetErrorMessage();

    mitkThrow() << errorMessage;
  }
  if (m_NavigationToolStorage->GetToolCount() < 1)
  {
    mitkThrow() << "No tracker tools available";
  }
  m_ExpectedNumberOfTools = m_NavigationToolStorage->GetToolCount();

  // Make sure we DONT display the surfaces that MITK uses for each tool.
  // We just want tracker matrix to be updated.
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
}


//-----------------------------------------------------------------------------
IGITracker::~IGITracker()
{
  if (m_TrackingVolumeNode.IsNotNull() && m_DataStorage->Exists(m_TrackingVolumeNode))
  {
    m_DataStorage->Remove(m_TrackingVolumeNode);
  }
}


//-----------------------------------------------------------------------------
void IGITracker::SetVisibilityOfTrackingVolume(bool isVisible)
{
  if (m_TrackingVolumeNode.IsNotNull())
  {
    m_TrackingVolumeNode->SetBoolProperty("visible", isVisible);
  }
}


//-----------------------------------------------------------------------------
bool IGITracker::GetVisibilityOfTrackingVolume() const
{
  bool result = false;
  if (m_TrackingVolumeNode.IsNotNull())
  {
    m_TrackingVolumeNode->GetBoolProperty("visible", result);
  }
  return result;
}


//-----------------------------------------------------------------------------
std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > IGITracker::GetTrackingDataAsMatrices()
{
  std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > result;
  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > data = this->GetTrackingData();
  std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >::const_iterator iter;
  for (iter = data.begin(); iter != data.end(); ++iter)
  {
    vtkSmartPointer<vtkMatrix4x4> mat = vtkSmartPointer<vtkMatrix4x4>::New();
    niftk::ConvertRotationAndTranslationToMatrix((*iter).second.first, (*iter).second.second, *mat);
    result.insert(std::pair<std::string, vtkSmartPointer<vtkMatrix4x4> >((*iter).first, mat));
  }
  return result;
}

} // end namespace
