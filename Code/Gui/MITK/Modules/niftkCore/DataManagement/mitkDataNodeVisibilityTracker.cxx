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
DataNodeVisibilityTracker::DataNodeVisibilityTracker(const mitk::DataStorage::Pointer dataStorage)
: mitk::DataNodePropertyListener(dataStorage, "visible")
, m_TrackedRenderer(0)
{
}


//-----------------------------------------------------------------------------
DataNodeVisibilityTracker::~DataNodeVisibilityTracker()
{
  this->SetTrackedRenderer(0);
}


//-----------------------------------------------------------------------------
void DataNodeVisibilityTracker::SetTrackedRenderer(const mitk::BaseRenderer* trackedRenderer)
{
  /// Note:
  /// We must not manage a renderer that we track, otherwise infinite recursion occurs.
  if (std::find(m_ManagedRenderers.begin(), m_ManagedRenderers.end(), trackedRenderer) != m_ManagedRenderers.end())
  {
    assert(false);
    return;
  }

  /// Note:
  /// Deliberately not checking if the new renderer to track is the same as the old one.
  /// This function can be used to reinitialise the visibility of every node.

  m_TrackedRenderer = trackedRenderer;

  mitk::DataStorage::SetOfObjects::ConstPointer all = this->GetDataStorage()->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    mitk::DataNode::Pointer node = it->Value();

    bool visibility = false;
    if (m_TrackedRenderer && !this->IsIgnored(node) && !node->GetProperty("renderer"))
    {
      /// TODO
      /// The const_cast is needed because of the MITK bug 17778. It should be removed after the bug is fixed.
      visibility = node->IsVisible(const_cast<mitk::BaseRenderer*>(m_TrackedRenderer));
    }

    for (std::size_t i = 0; i < m_ManagedRenderers.size(); ++i)
    {
      if (!this->IsIgnored(node) && !node->GetProperty("renderer"))
      {
        /// TODO
        /// The const_cast is needed because of the MITK bug 17778. It should be removed after the bug is fixed.
        node->SetBoolProperty("visible", visibility, const_cast<mitk::BaseRenderer*>(m_ManagedRenderers[i]));
      }
    }
  }

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


//-----------------------------------------------------------------------------
void DataNodeVisibilityTracker::SetManagedRenderers(const std::vector<const mitk::BaseRenderer*>& managedRenderers)
{
  /// Note:
  /// We must not manage a renderer that we track, otherwise infinite recursion occurs.
  bool hasNull = std::find(managedRenderers.begin(), managedRenderers.end(), (const mitk::BaseRenderer*)0) != managedRenderers.end();
  bool hasTrackedRenderer = std::find(managedRenderers.begin(), managedRenderers.end(), m_TrackedRenderer) != managedRenderers.end();
  if (hasNull || hasTrackedRenderer)
  {
    assert(false);
    return;
  }

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
void DataNodeVisibilityTracker::OnNodeAdded(mitk::DataNode* node)
{
  mitk::BoolProperty* globalProperty = dynamic_cast<mitk::BoolProperty*>(node->GetProperty("visible", 0));

  bool visibility = false;
  if (m_TrackedRenderer && !this->IsIgnored(node) && !node->GetProperty("renderer"))
  {
    /// TODO
    /// The const_cast is needed because of the MITK bug 17778. It should be removed after the bug is fixed.
    visibility = node->IsVisible(const_cast<mitk::BaseRenderer*>(m_TrackedRenderer));
  }

  for (unsigned int i = 0; i < m_ManagedRenderers.size(); i++)
  {
    mitk::BoolProperty* rendererSpecificProperty = dynamic_cast<mitk::BoolProperty*>(node->GetProperty("visible", m_ManagedRenderers[i]));

    /// Note:
    /// GetProperty() returns the global property if there is no renderer specific one.
    /// If there is no renderer specific property then we create one and set to the same
    /// visibility as in the tracked renderer. Otherwise, we leave it as it is.
    if (rendererSpecificProperty == globalProperty)
    {
      /// TODO
      /// The const_cast is needed because of the MITK bug 17778. It should be removed after the bug is fixed.
      node->SetBoolProperty("visible", visibility, const_cast<mitk::BaseRenderer*>(m_ManagedRenderers[i]));
    }
  }

  /// Register the observers so that OnPropertyChanged is called when the visibility changes.
  Superclass::OnNodeAdded(node);
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
}

}
