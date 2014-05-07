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
DataStorageVisibilityTracker::DataStorageVisibilityTracker()
: m_DataStorage(0)
, m_TrackedRenderer(0)
, m_Listener(0)
{
  this->Init(0);
}


//-----------------------------------------------------------------------------
DataStorageVisibilityTracker::DataStorageVisibilityTracker(const mitk::DataStorage::Pointer dataStorage)
: m_DataStorage(0)
, m_TrackedRenderer(0)
, m_Listener(0)
{
 this->Init(dataStorage);
}


//-----------------------------------------------------------------------------
DataStorageVisibilityTracker::~DataStorageVisibilityTracker()
{
  this->SetTrackedRenderer(0);

  m_Listener->PropertyChanged += mitk::MessageDelegate2<DataStorageVisibilityTracker, mitk::DataNode*, mitk::BaseRenderer*>(this, &DataStorageVisibilityTracker::OnPropertyChanged);
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::Init(const mitk::DataStorage::Pointer dataStorage)
{
  m_Listener = mitk::DataStoragePropertyListener::New();
  m_Listener->SetPropertyName("visible");
  m_Listener->SetDataStorage(dataStorage);

  m_Listener->PropertyChanged += mitk::MessageDelegate2<DataStorageVisibilityTracker, mitk::DataNode*, mitk::BaseRenderer*>(this, &DataStorageVisibilityTracker::OnPropertyChanged);
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::SetDataStorage(const mitk::DataStorage::Pointer dataStorage)
{
  m_DataStorage = dataStorage;
  m_Listener->SetDataStorage(dataStorage);
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::SetTrackedRenderer(mitk::BaseRenderer* trackedRenderer)
{
  if (trackedRenderer != m_TrackedRenderer)
  {
    if (m_TrackedRenderer)
    {
      /// Disable visibility of all nodes in the managed render windows.

      assert(m_DataStorage.IsNotNull());

      mitk::DataStorage::SetOfObjects::ConstPointer all = m_DataStorage->GetAll();
      for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
      {
        mitk::DataNode::Pointer node = it->Value();

        if (this->IsIgnored(node))
        {
          continue;
        }

        for (std::size_t i = 0; i < m_ManagedRenderers.size(); ++i)
        {
          mitk::BoolProperty* property = dynamic_cast<mitk::BoolProperty*>(node->GetProperty("visible", m_ManagedRenderers[i]));
          if (property && property->GetValue())
          {
            property->SetValue(false);
          }
        }
      }
    }

    m_TrackedRenderer = trackedRenderer;

    if (trackedRenderer)
    {
      std::vector<mitk::BaseRenderer*> renderersToTrack(1);
      renderersToTrack[0] = trackedRenderer;
      m_Listener->SetRenderers(renderersToTrack);
    }
    else
    {
      std::vector<mitk::BaseRenderer*> renderersToTrack(0);
      m_Listener->SetRenderers(renderersToTrack);
    }

    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::SetManagedRenderers(const std::vector<mitk::BaseRenderer*>& managedRenderers)
{
  m_ManagedRenderers = managedRenderers;
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
  return std::find(m_NodesToIgnore.begin(), m_NodesToIgnore.end(), node) != m_NodesToIgnore.end();
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::OnPropertyChanged(mitk::DataNode* node, mitk::BaseRenderer* renderer)
{
  /// We do not have anything to do if:
  ///   - the data storage has not been initialised
  ///   - there is no tracked or managed renderer
  ///   - the node is added to the ignore list
  ///   - a renderer specific visibility has changed for a different renderer than which we track.
  if (m_DataStorage.IsNull()
      || !m_TrackedRenderer
      || m_ManagedRenderers.empty()
      || this->IsIgnored(node)
      || (renderer && renderer != m_TrackedRenderer))
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

  bool trackedWindowVisible = false;
  bool foundTrackedWindowVisible = node->GetBoolProperty("visible", trackedWindowVisible, m_TrackedRenderer);

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
  for (std::size_t i = 0; i < m_ManagedRenderers.size(); ++i)
  {
    node->SetBoolProperty("visible", finalVisibility, m_ManagedRenderers[i]);
  }

  // don't forget to unblock.
  m_Listener->SetBlocked(wasBlocked);
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::NotifyAll()
{
  m_Listener->NotifyAll();
}

} // end namespace
