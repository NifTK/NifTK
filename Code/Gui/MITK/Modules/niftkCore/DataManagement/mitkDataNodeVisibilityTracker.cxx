/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDataNodeVisibilityTracker.h"
#include <mitkBaseRenderer.h>

namespace mitk
{

//-----------------------------------------------------------------------------
DataNodeVisibilityTracker::DataNodeVisibilityTracker()
: mitk::DataNodePropertyListener("visible")
, m_TrackedRenderer(0)
{
}


//-----------------------------------------------------------------------------
DataNodeVisibilityTracker::~DataNodeVisibilityTracker()
{
  this->SetTrackedRenderer(0);
}


//-----------------------------------------------------------------------------
bool DataNodeVisibilityTracker::GetDefaultVisibility() const
{
  return m_DefaultVisibility;
}


//-----------------------------------------------------------------------------
void DataNodeVisibilityTracker::SetDefaultVisibility(bool defaultVisibility)
{
  m_DefaultVisibility = defaultVisibility;
}


//-----------------------------------------------------------------------------
void DataNodeVisibilityTracker::SetTrackedRenderer(const mitk::BaseRenderer* trackedRenderer)
{
  if (trackedRenderer != m_TrackedRenderer)
  {
    if (m_TrackedRenderer)
    {
      /// Disable visibility of all nodes in the managed render windows.

      assert(this->GetDataStorage().IsNotNull());

      // block the calls, so we can update stuff, without repeated callback loops.
      bool wasBlocked = this->SetBlocked(true);

      mitk::DataStorage::SetOfObjects::ConstPointer all = this->GetDataStorage()->GetAll();
      for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
      {
        mitk::DataNode::Pointer node = it->Value();

        if (this->IsIgnored(node))
        {
          continue;
        }

        mitk::BoolProperty* globalProperty = dynamic_cast<mitk::BoolProperty*>(node->GetProperty("visible", 0));
        for (std::size_t i = 0; i < m_ManagedRenderers.size(); ++i)
        {
          /// Note:
          /// GetProperty() returns the global property if there is no renderer specific one.
          /// If there is no renderer specific property then we create one and set to false.
          /// Otherwise, we set it to false, unless it alread was.
          mitk::BoolProperty* rendererSpecificProperty = dynamic_cast<mitk::BoolProperty*>(node->GetProperty("visible", m_ManagedRenderers[i]));
          if (rendererSpecificProperty == globalProperty)
          {
            /// TODO
            /// The const_cast is needed because of the MITK bug 17778. It should be removed after the bug is fixed.
            node->SetBoolProperty("visible", false, const_cast<mitk::BaseRenderer*>(m_ManagedRenderers[i]));
          }
          else if (rendererSpecificProperty && rendererSpecificProperty->GetValue())
          {
            rendererSpecificProperty->SetValue(false);
          }
        }
      }

      this->SetBlocked(wasBlocked);
    }

    m_TrackedRenderer = trackedRenderer;

    if (trackedRenderer)
    {
      std::vector<const mitk::BaseRenderer*> renderersToTrack(1);
      renderersToTrack[0] = trackedRenderer;
      this->SetRenderers(renderersToTrack);
    }
    else
    {
      std::vector<const mitk::BaseRenderer*> renderersToTrack(0);
      this->SetRenderers(renderersToTrack);
    }
  }
}


//-----------------------------------------------------------------------------
void DataNodeVisibilityTracker::SetManagedRenderers(const std::vector<const mitk::BaseRenderer*>& managedRenderers)
{
  m_ManagedRenderers = managedRenderers;
}


//-----------------------------------------------------------------------------
void DataNodeVisibilityTracker::SetNodesToIgnore(const std::vector<mitk::DataNode*>& nodesToIgnore)
{
  m_NodesToIgnore = nodesToIgnore;
}


//-----------------------------------------------------------------------------
bool DataNodeVisibilityTracker::IsIgnored(mitk::DataNode* node)
{
  return std::find(m_NodesToIgnore.begin(), m_NodesToIgnore.end(), node) != m_NodesToIgnore.end();
}


//-----------------------------------------------------------------------------
void DataNodeVisibilityTracker::NodeAdded(mitk::DataNode* node)
{
  if (m_ManagedRenderers.size() > 0)
  {
    /// TODO
//    node->SetBoolProperty("visible", m_Visibility);
    for (unsigned int i = 0; i < m_ManagedRenderers.size(); i++)
    {
      /// TODO
      /// The const_cast is needed because of the MITK bug 17778. It should be removed after the bug is fixed.
      node->SetBoolProperty("visible", m_DefaultVisibility, const_cast<mitk::BaseRenderer*>(m_ManagedRenderers[i]));
    }
  }
  else
  {
    node->SetBoolProperty("visible", m_DefaultVisibility);
  }
}


//-----------------------------------------------------------------------------
void DataNodeVisibilityTracker::OnPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  /// We do not have anything to do if:
  ///   - the data storage has not been initialised
  ///   - there is no tracked or managed renderer
  ///   - the node is added to the ignore list
  ///   - a renderer specific visibility has changed for a different renderer than which we track.
  if (this->GetDataStorage().IsNull()
      || !m_TrackedRenderer
      || m_ManagedRenderers.empty()
      || this->IsIgnored(node)
      || (renderer && renderer != m_TrackedRenderer))
  {
    return;
  }

  // block the calls, so we can update stuff, without repeated callback loops.
  bool wasBlocked = this->SetBlocked(true);

  // Intention : This object should display all the data nodes visible in the focused window, and none others.
  // Assumption: Renderer specific properties override the global ones.
  // so......    Objects will be visible, unless the the node has a render window specific property that says otherwise.

  bool globalVisible = false;
  bool foundGlobalVisible = node->GetBoolProperty("visible", globalVisible);

  bool trackedWindowVisible = false;
  /// TODO
  /// The const_cast is needed because of the MITK bug 17778. It should be removed after the bug is fixed.
  bool foundTrackedWindowVisible = node->GetBoolProperty("visible", trackedWindowVisible, const_cast<mitk::BaseRenderer*>(m_TrackedRenderer));

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
    /// TODO
    /// The const_cast is needed because of the MITK bug 17778. It should be removed after the bug is fixed.
    node->SetBoolProperty("visible", finalVisibility, const_cast<mitk::BaseRenderer*>(m_ManagedRenderers[i]));
  }

  // don't forget to unblock.
  this->SetBlocked(wasBlocked);
}

}
