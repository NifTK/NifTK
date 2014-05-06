/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDataStorageVisibilityTracker.h"
#include <mitkBaseRenderer.h>

namespace mitk
{

//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::Init(const mitk::DataStorage::Pointer dataStorage)
{
  m_Listener = mitk::DataStoragePropertyListener::New();
  m_Listener->SetPropertyName("visible");
  m_Listener->SetDataStorage(dataStorage);

  m_RenderersToTrack.clear();
  m_RenderersToUpdate.clear();
  m_NodesToIgnore.clear();

  m_Listener->PropertyChanged += mitk::MessageDelegate2<DataStorageVisibilityTracker, mitk::DataNode*, mitk::BaseRenderer*>(this, &DataStorageVisibilityTracker::OnPropertyChanged);
}


//-----------------------------------------------------------------------------
DataStorageVisibilityTracker::DataStorageVisibilityTracker()
: m_Listener(NULL)
, m_DataStorage(NULL)
{
  this->Init(NULL);
}


//-----------------------------------------------------------------------------
DataStorageVisibilityTracker::DataStorageVisibilityTracker(const mitk::DataStorage::Pointer dataStorage)
: m_Listener(NULL)
, m_DataStorage(NULL)
{
 this->Init(dataStorage);
}


//-----------------------------------------------------------------------------
DataStorageVisibilityTracker::~DataStorageVisibilityTracker()
{
  m_Listener->PropertyChanged += mitk::MessageDelegate2<DataStorageVisibilityTracker, mitk::DataNode*, mitk::BaseRenderer*>(this, &DataStorageVisibilityTracker::OnPropertyChanged);
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::SetRenderersToUpdate(const std::vector<mitk::BaseRenderer*>& renderersToUpdate)
{
  m_RenderersToUpdate = renderersToUpdate;
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::SetRenderersToTrack(const std::vector<mitk::BaseRenderer*>& renderersToTrack)
{
  m_RenderersToTrack = renderersToTrack;
  m_Listener->SetRenderers(renderersToTrack);
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::SetDataStorage(const mitk::DataStorage::Pointer dataStorage)
{
  m_DataStorage = dataStorage;
  m_Listener->SetDataStorage(dataStorage);
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::SetNodesToIgnore(const std::vector<mitk::DataNode*>& nodesToIgnore)
{
  m_NodesToIgnore = nodesToIgnore;
}


//-----------------------------------------------------------------------------
bool DataStorageVisibilityTracker::IsIgnored(mitk::DataNode* node)
{
  bool isExcluded = false;

  std::vector<mitk::DataNode*>::iterator iter;
  for (iter = m_NodesToIgnore.begin(); iter != m_NodesToIgnore.end(); iter++)
  {
    if (*iter == node)
    {
      isExcluded = true;
      break;
    }
  }
  return isExcluded;
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::NotifyAll()
{
  m_Listener->NotifyAll();
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::OnPropertyChanged(mitk::DataNode* node, mitk::BaseRenderer* renderer)
{
  if (m_DataStorage.IsNull() || m_RenderersToTrack.empty() || m_RenderersToUpdate.empty())
  {
    return;
  }

  if (this->IsIgnored(node))
  {
    return;
  }

  // block the calls, so we can update stuff, without repeated callback loops.
  bool wasBlocked = m_Listener->GetBlocked();
  m_Listener->SetBlocked(true);

  // Intention : This object should display all the data nodes visible in the focused window, and none others.
  // Assumption: Renderer specific properties override the global ones.
  // so......    Objects will be visible, unless the the node has a render window specific property that says otherwise.

  bool globalVisible = false;
  bool foundGlobalVisible = node->GetBoolProperty("visible", globalVisible);

  for (std::size_t i = 0; i < m_RenderersToTrack.size(); ++i)
  {
    bool trackedWindowVisible = false;
    bool foundTrackedWindowVisible = node->GetBoolProperty("visible", trackedWindowVisible, m_RenderersToTrack[i]);

    // We default to ON.
    bool finalVisibility = true;

    // The logic.
    if ((foundTrackedWindowVisible && !trackedWindowVisible)
        || (foundGlobalVisible && !globalVisible)
        )
    {
      finalVisibility = false;
    }

    // Set the final visibility flag
    for (std::size_t j = 0; j < m_RenderersToUpdate.size(); ++j)
    {
      node->SetBoolProperty("visible", finalVisibility, m_RenderersToUpdate[j]);
    }
  }

  // don't forget to unblock.
  m_Listener->SetBlocked(wasBlocked);
}

} // end namespace
