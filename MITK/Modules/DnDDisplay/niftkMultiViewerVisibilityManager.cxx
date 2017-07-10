/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMultiViewerVisibilityManager.h"

#include <itkConversionUtils.h>
#include <itkSpatialOrientationAdapter.h>

#include <mitkVtkResliceInterpolationProperty.h>
#include <mitkImageAccessByItk.h>

#include <niftkDataStorageUtils.h>

#include "niftkSingleViewerWidget.h"

namespace niftk
{

//-----------------------------------------------------------------------------
MultiViewerVisibilityManager::MultiViewerVisibilityManager(mitk::DataStorage::Pointer dataStorage)
  : niftk::DataNodePropertyListener(dataStorage, "visible"),
    m_DropType(DNDDISPLAY_DROP_SINGLE),
    m_DefaultWindowLayout(WINDOW_LAYOUT_CORONAL),
    m_InterpolationType(DNDDISPLAY_CUBIC_INTERPOLATION),
    m_Accumulate(false),
    m_VisibilityBinding(false),
    m_VisibilityOfForeignNodesLocked(true)
{
  bool wasBlocked = this->SetBlocked(true);
  mitk::DataStorage::SetOfObjects::ConstPointer all = this->GetDataStorage()->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    mitk::DataNode::Pointer node = it->Value();
    node->SetVisibility(false);
  }
  this->SetBlocked(wasBlocked);
}


//-----------------------------------------------------------------------------
MultiViewerVisibilityManager::~MultiViewerVisibilityManager()
{
}


//-----------------------------------------------------------------------------
DnDDisplayDropType MultiViewerVisibilityManager::GetDropType() const
{
  return m_DropType;
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::SetDropType(DnDDisplayDropType dropType)
{
  m_DropType = dropType;
  this->SetVisibilityBinding(dropType == DNDDISPLAY_DROP_ALL);
}


//-----------------------------------------------------------------------------
DnDDisplayInterpolationType MultiViewerVisibilityManager::GetInterpolationType() const
{
  return m_InterpolationType;
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::SetInterpolationType(DnDDisplayInterpolationType interpolationType)
{
  m_InterpolationType = interpolationType;
}


//-----------------------------------------------------------------------------
WindowLayout MultiViewerVisibilityManager::GetDefaultWindowLayout() const
{
  return m_DefaultWindowLayout;
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::SetDefaultWindowLayout(WindowLayout defaultWindowLayout)
{
  m_DefaultWindowLayout = defaultWindowLayout;
}


//-----------------------------------------------------------------------------
bool MultiViewerVisibilityManager::GetAccumulateWhenDropping() const
{
  return m_Accumulate;
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::SetAccumulateWhenDropping(bool accumulate)
{
  m_Accumulate = accumulate;
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::RegisterViewer(SingleViewerWidget* viewer)
{
  m_DroppedNodes[viewer] = std::set<mitk::DataNode*>();
  m_Viewers.push_back(viewer);

  std::vector<mitk::DataNode*> nodes;

  mitk::DataStorage::SetOfObjects::ConstPointer all = this->GetDataStorage()->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    /// We set the renderer specific visibility of the nodes that have global visibility property.
    /// (Regardless if they are visible globally or not.)
    if (it->Value()->GetProperty("visible"))
    {
      nodes.push_back(it->Value());
    }
  }

  viewer->SetVisibility(nodes, false);

  this->connect(viewer, SIGNAL(NodesDropped(const std::vector<mitk::DataNode*>&)), SLOT(OnNodesDropped(const std::vector<mitk::DataNode*>&)));
  this->connect(viewer, SIGNAL(WindowSelected()), SLOT(OnWindowSelected()));
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::DeregisterViewers(std::size_t startIndex, std::size_t endIndex)
{
  if (endIndex == -1)
  {
    endIndex = m_Viewers.size();
  }
  for (std::size_t i = startIndex; i < endIndex; ++i)
  {
    SingleViewerWidget* viewer = m_Viewers[i];
    QObject::disconnect(viewer, SIGNAL(NodesDropped(const std::vector<mitk::DataNode*>&)), this, SLOT(OnNodesDropped(const std::vector<mitk::DataNode*>&)));
    QObject::disconnect(viewer, SIGNAL(WindowSelected()), this, SLOT(OnWindowSelected()));
    this->RemoveNodesFromViewer(viewer);
    m_DroppedNodes.erase(viewer);
  }
  m_Viewers.erase(m_Viewers.begin() + startIndex, m_Viewers.begin() + endIndex);
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::ClearViewers(std::size_t startIndex, std::size_t endIndex)
{
  if (endIndex == -1)
  {
    endIndex = m_Viewers.size();
  }
  for (std::size_t i = startIndex; i < endIndex; i++)
  {
    this->RemoveNodesFromViewer(m_Viewers[i]);
  }
}


//-----------------------------------------------------------------------------
bool MultiViewerVisibilityManager::IsVisibilityBound() const
{
  return m_VisibilityBinding;
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::SetVisibilityBinding(bool bound)
{
  if (bound != m_VisibilityBinding)
  {
    m_VisibilityBinding = bound;
    emit VisibilityBindingChanged(bound);
  }
}


//-----------------------------------------------------------------------------
bool MultiViewerVisibilityManager::IsVisibilityOfForeignNodesLocked() const
{
  return m_VisibilityOfForeignNodesLocked;
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::SetVisibilityOfForeignNodesLocked(bool locked)
{
  if (locked != m_VisibilityOfForeignNodesLocked)
  {
    m_VisibilityOfForeignNodesLocked = locked;
  }
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::OnNodeAdded(mitk::DataNode* node)
{
  /// Note:
  /// Do not manage the visibility of the crosshair planes.
  if (node->GetProperty("renderer"))
  {
    return;
  }

  /// When a new node is added, the local visibility has to be set explicitely
  /// for each viewer, even if it is the same as the global visibility.
  ///
  /// The policy is the following:
  ///
  ///   - If the global visibility is 'true' but the node has *not* been dropped onto
  ///     the selected viewer (neither itself nor any of its sources), we change the
  ///     global visibility to 'false'. This switches off the visibility check box of
  ///     the top level data node in the Data Manager, e.g. after File/Open.
  ///
  ///   - Exceptions from the rule above are only "helper" nodes. We do not force the
  ///     global visibility of helper nodes to be off even if they are 'foreign' nodes.
  ///
  ///   - The local visibility in the selected viewer is set to the same as the global
  ///     visibility. Note that the global visibility may have been cleared at the
  ///     first point.
  ///
  ///   - If the visibility is bound across the viewers, the local visibility in the
  ///     other viewers is set to the same as in the selected viewer.
  ///
  ///   - If the visibility is *not* bound across the viewers, the local visibility
  ///     in the other viewers is set to false.

  bool isHelperObject = false;
  node->GetBoolProperty("helper object", isHelperObject);

  bool globalVisibility = node->IsVisible(nullptr);

  SingleViewerWidget* selectedViewer = nullptr;
  for (auto viewer: m_Viewers)
  {
    if (viewer->IsFocused())
    {
      selectedViewer = viewer;
      break;
    }
  }

  if (!isHelperObject && globalVisibility && this->IsForeignNode(node, selectedViewer))
  {
    globalVisibility = false;
    bool wasBlocked = this->SetBlocked(true);
    node->SetVisibility(false);
    this->SetBlocked(wasBlocked);
  }

  std::vector<mitk::DataNode*> nodes(1);
  nodes[0] = node;
  for (auto viewer: m_Viewers)
  {
    bool localVisibility = (viewer == selectedViewer || m_VisibilityBinding) ? globalVisibility : false;
    viewer->SetVisibility(nodes, localVisibility);
  }

  mitk::VtkResliceInterpolationProperty* interpolationProperty =
      dynamic_cast<mitk::VtkResliceInterpolationProperty*>(node->GetProperty("reslice interpolation"));
  if (interpolationProperty)
  {
    if (m_InterpolationType == DNDDISPLAY_NO_INTERPOLATION)
    {
      interpolationProperty->SetInterpolationToNearest();
    }
    else if (m_InterpolationType == DNDDISPLAY_LINEAR_INTERPOLATION)
    {
      interpolationProperty->SetInterpolationToLinear();
    }
    else if (m_InterpolationType == DNDDISPLAY_CUBIC_INTERPOLATION)
    {
      interpolationProperty->SetInterpolationToCubic();
    }
  }

  Superclass::OnNodeAdded(node);
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::OnNodeRemoved(mitk::DataNode* node)
{
  Superclass::OnNodeRemoved(node);

  // This is just to trigger updating the intensity annotations.
  for (auto viewer: m_Viewers)
  {
    if (viewer->IsFocused())
    {
      std::vector<mitk::DataNode*> nodes(1);
      nodes[0] = node;
      viewer->SetVisibility(nodes, false);
    }
  }
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::OnPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  /// Note:
  /// The renderer must be nullptr because we are listening to the global visibility only.
  assert(renderer == nullptr);

  bool globalVisibility = node->IsVisible(nullptr);

  SingleViewerWidget* selectedViewer = nullptr;
  for (auto viewer: m_Viewers)
  {
    if (viewer->IsFocused())
    {
      selectedViewer = viewer;
      break;
    }
  }

  /// A node is 'foreign' to a viewer if neither itself nor any of its source nodes
  /// have not been dropped on the viewer. If the node is foreign to the selected
  /// viewer and the visibility of foreign nodes is locked, we need to 'undo' the
  /// property change.

  if (selectedViewer == nullptr
      || (this->IsForeignNode(node, selectedViewer)
          && m_VisibilityOfForeignNodesLocked))
  {
    bool wasBlocked = this->SetBlocked(true);
    node->SetVisibility(!globalVisibility);
    this->SetBlocked(wasBlocked);
    return;
  }

  /// Otherwise, we set the local visibility to the new global visibility
  /// in the selected viewer, and if the visibility is bound across the
  /// viewers then in the other viewers as well.
  std::vector<mitk::DataNode*> nodes(1);
  nodes[0] = node;
  for (auto viewer: m_Viewers)
  {
    if (viewer == selectedViewer || m_VisibilityBinding)
    {
      viewer->SetVisibility(nodes, globalVisibility);
    }
  }
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::RemoveNodesFromViewer(SingleViewerWidget* viewer)
{
  assert(viewer);

  mitk::DataStorage::SetOfObjects::ConstPointer allNodes = this->GetDataStorage()->GetAll();
  std::vector<mitk::DataNode*> nodes;
  for (auto it = allNodes->Begin(); it != allNodes->End(); ++it)
  {
    mitk::DataNode::Pointer node = it->Value();
    nodes.push_back(node.GetPointer());
  }
  viewer->SetVisibility(nodes, false);

  m_DroppedNodes[viewer].clear();
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::AddNodeToViewer(SingleViewerWidget* viewer, mitk::DataNode* node)
{
  assert(viewer);

  std::vector<mitk::DataNode*> nodes;
  nodes.push_back(node);

  bool wasBlocked = this->SetBlocked(true);

  node->SetVisibility(true);
  m_DroppedNodes[viewer].insert(node);

  mitk::DataStorage::SetOfObjects::ConstPointer derivedNodes = this->GetDataStorage()->GetDerivations(node, nullptr, false);
  for (auto it = derivedNodes->Begin(); it != derivedNodes->End(); ++it)
  {
    mitk::DataNode* derivedNode = it->Value();
    nodes.push_back(derivedNode);
    derivedNode->SetVisibility(true);
  }

  this->SetBlocked(wasBlocked);

  viewer->ApplyGlobalVisibility(nodes);
}


//-----------------------------------------------------------------------------
bool MultiViewerVisibilityManager::IsForeignNode(mitk::DataNode* node, SingleViewerWidget* viewer)
{
  auto& nodesDroppedOnViewer = m_DroppedNodes[viewer];

  if (nodesDroppedOnViewer.find(node) != nodesDroppedOnViewer.end())
  {
    return false;
  }

  mitk::DataStorage* dataStorage = this->GetDataStorage();
  mitk::DataStorage::SetOfObjects::ConstPointer sources = dataStorage->GetSources(node, nullptr, false);
  for (auto it = sources->Begin(); it != sources->End(); ++it)
  {
    mitk::DataNode::Pointer source = it->Value();
    if (nodesDroppedOnViewer.find(source) != nodesDroppedOnViewer.end())
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------------
mitk::TimeGeometry::Pointer MultiViewerVisibilityManager::GetTimeGeometry(std::vector<mitk::DataNode*> nodes, int nodeIndex)
{
  mitk::TimeGeometry::Pointer geometry;
  int indexThatWeActuallyUsed = -1;

  // If nodeIndex < 0, we are choosing the best geometry from all available nodes.
  if (nodeIndex < 0)
  {
    // First try to find an image geometry, and if so, use the first one.
    for (std::size_t i = 0; i < nodes.size(); i++)
    {
      mitk::Image* image = dynamic_cast<mitk::Image*>(nodes[i]->GetData());
      if (image)
      {
        geometry = image->GetTimeGeometry();
        indexThatWeActuallyUsed = i;
        break;
      }
    }

    // Failing that, use the first geometry available.
    if (geometry.IsNull())
    {
      for (std::size_t i = 0; i < nodes.size(); i++)
      {
        mitk::BaseData* data = nodes[i]->GetData();
        if (data)
        {
          geometry = data->GetTimeGeometry();
          indexThatWeActuallyUsed = i;
          break;
        }
      }
    }
  }
  // So, the caller has nominated a specific node, lets just use that one.
  else if (nodeIndex >= 0 && nodeIndex < (int)nodes.size())
  {
    mitk::BaseData* data = nodes[nodeIndex]->GetData();
    if (data)
    {
      geometry = data->GetTimeGeometry();
      indexThatWeActuallyUsed = nodeIndex;
    }
  }
  // Essentially, the nodeIndex is garbage, so just pick the first one.
  else
  {
    mitk::BaseData* data = nodes[0]->GetData();
    if (data)
    {
      geometry = data->GetTimeGeometry();
      indexThatWeActuallyUsed = 0;
    }
  }

  // In addition, if the node is NOT a greyscale image, we try and search the parents
  // of the node to find a greyscale image and use that one in preference.
  // This assumes that derived datasets, such as point sets, surfaces, segmented
  // volumes are correctly assigned to parents.
  if (indexThatWeActuallyUsed != -1)
  {
    mitk::DataNode* nodeThatWeActuallyUsed = nodes[indexThatWeActuallyUsed];
    if (!niftk::IsNodeANonBinaryImage(nodeThatWeActuallyUsed))
    {
      mitk::DataNode* node = niftk::FindFirstParentImage(this->GetDataStorage(), nodeThatWeActuallyUsed, false);
      if (node)
      {
        mitk::BaseData* data = node->GetData();
        if (data)
        {
          geometry = data->GetTimeGeometry();
        }
      }
    }
  }

  return geometry;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MultiViewerVisibilityManager::GetAsAcquiredOrientation(itk::Image<TPixel, VImageDimension>* itkImage, WindowOrientation& outputOrientation)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;

  typename itk::SpatialOrientationAdapter adaptor;
  typename itk::SpatialOrientation::ValidCoordinateOrientationFlags orientation;
  orientation = adaptor.FromDirectionCosines(itkImage->GetDirection());
  std::string orientationString = itk::ConvertSpatialOrientationToString(orientation);

  if (orientationString[0] == 'L' || orientationString[0] == 'R')
  {
    if (orientationString[1] == 'A' || orientationString[1] == 'P')
    {
      outputOrientation = WINDOW_ORIENTATION_AXIAL;
    }
    else
    {
      outputOrientation = WINDOW_ORIENTATION_CORONAL;
    }
  }
  else if (orientationString[0] == 'A' || orientationString[0] == 'P')
  {
    if (orientationString[1] == 'L' || orientationString[1] == 'R')
    {
      outputOrientation = WINDOW_ORIENTATION_AXIAL;
    }
    else
    {
      outputOrientation = WINDOW_ORIENTATION_SAGITTAL;
    }
  }
  else if (orientationString[0] == 'S' || orientationString[0] == 'I')
  {
    if (orientationString[1] == 'L' || orientationString[1] == 'R')
    {
      outputOrientation = WINDOW_ORIENTATION_CORONAL;
    }
    else
    {
      outputOrientation = WINDOW_ORIENTATION_SAGITTAL;
    }
  }
}


//-----------------------------------------------------------------------------
WindowLayout MultiViewerVisibilityManager::GetWindowLayout(std::vector<mitk::DataNode*> nodes)
{
  WindowLayout windowLayout = m_DefaultWindowLayout;
  if (windowLayout == WINDOW_LAYOUT_AS_ACQUIRED)
  {
    // "As Acquired" means you take the orientation of the XY plane
    // in the original image data, so we switch to ITK to work it out.
    WindowOrientation orientation = WINDOW_ORIENTATION_CORONAL;

    mitk::Image::Pointer image;
    for (std::size_t i = 0; i < nodes.size(); i++)
    {
      image = dynamic_cast<mitk::Image*>(nodes[i]->GetData());
      if (image.IsNotNull())
      {
        break;
      }
    }
    if (image.IsNotNull() && image->GetDimension() >= 3)
    {
      try
      {
        AccessFixedDimensionByItk_n(image, GetAsAcquiredOrientation, 3, (orientation));
      }
      catch (const mitk::AccessByItkException &e)
      {
        MITK_ERROR << "MultiViewerVisibilityManager::GetWindowLayout() failed to work out 'As Acquired' orientation." << e.what() << std::endl;
      }
    }
    else
    {
      MITK_ERROR << "MultiViewerVisibilityManager::GetWindowLayout() failed to find an image to work out 'As Acquired' orientation." << std::endl;
    }

    if (orientation == WINDOW_ORIENTATION_AXIAL)
    {
      windowLayout = WINDOW_LAYOUT_AXIAL;
    }
    else if (orientation == WINDOW_ORIENTATION_SAGITTAL)
    {
      windowLayout = WINDOW_LAYOUT_SAGITTAL;
    }
    else if (orientation == WINDOW_ORIENTATION_CORONAL)
    {
      windowLayout = WINDOW_LAYOUT_CORONAL;
    }
    else
    {
      MITK_ERROR << "MultiViewerVisibilityManager::GetWindowLayout() defaulting to window layout " << windowLayout << std::endl;
    }
  }
  return windowLayout;
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::OnWindowSelected()
{
  SingleViewerWidget* selectedViewer = qobject_cast<SingleViewerWidget*>(QObject::sender());
  this->UpdateGlobalVisibilities(selectedViewer->GetSelectedRenderWindow()->GetRenderer());
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::UpdateGlobalVisibilities(mitk::BaseRenderer* renderer)
{
  bool wasBlocked = this->SetBlocked(true);
  mitk::DataStorage::SetOfObjects::ConstPointer all = this->GetDataStorage()->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    /// We set the global visibility of the nodes that the same as the renderer specific visibility.
    mitk::DataNode::Pointer node = it->Value();
    if (!node->GetProperty("renderer"))
    {
      bool visibility = node->IsVisible(renderer);
      node->SetVisibility(visibility);
    }
  }
  this->SetBlocked(wasBlocked);
}


//-----------------------------------------------------------------------------
void MultiViewerVisibilityManager::OnNodesDropped(const std::vector<mitk::DataNode*>& droppedNodes)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());

  int viewerIndex = std::find(m_Viewers.begin(), m_Viewers.end(), viewer) - m_Viewers.begin();
  assert(viewerIndex != m_Viewers.size());

  WindowLayout windowLayout = this->GetWindowLayout(droppedNodes);

  if (m_DropType == DNDDISPLAY_DROP_SINGLE)
  {
    mitk::TimeGeometry::Pointer timeGeometry = this->GetTimeGeometry(droppedNodes, -1);
    if (timeGeometry.IsNull())
    {
      MITK_ERROR << "Error, dropping " << droppedNodes.size() << " nodes into viewer " << viewerIndex
                 << ", could not find geometry which must be a programming bug.";
      return;
    }

    // Clear all nodes from the single viewer denoted by viewerIndex (the one that was dropped into).
    if (!this->GetAccumulateWhenDropping())
    {
      this->RemoveNodesFromViewer(viewer);
    }

    // Then set up geometry of that single viewer.
    if (!this->GetAccumulateWhenDropping())
    {
      viewer->SetTimeGeometry(timeGeometry.GetPointer());
      viewer->SetWindowLayout(windowLayout);
      viewer->SetEnabled(true);
    }

    // Then add all nodes into the same viewer denoted by viewerIndex (the one that was dropped into).
    for (std::size_t i = 0; i < droppedNodes.size(); i++)
    {
      this->AddNodeToViewer(viewer, droppedNodes[i]);
    }
  }
  else if (m_DropType == DNDDISPLAY_DROP_MULTIPLE)
  {
    // Work out which viewer we are actually dropping into.
    // We aim to put one object, in each of consecutive viewers.
    // If we hit the end (of the 5x5=25 list), we go back to zero.

    std::size_t dropIndex = viewerIndex;

    for (std::size_t i = 0; i < droppedNodes.size(); i++)
    {
      while (dropIndex < m_Viewers.size() && !m_Viewers[dropIndex]->isVisible())
      {
        // i.e. if the viewer we are in, is not visible, keep looking
        dropIndex++;
      }
      if (dropIndex == m_Viewers.size())
      {
        // give up? Or we could go back to zero?
        dropIndex = 0;
      }

      SingleViewerWidget* viewerDropInto = m_Viewers[dropIndex];

      mitk::TimeGeometry::Pointer timeGeometry = this->GetTimeGeometry(droppedNodes, i);
      if (timeGeometry.IsNull())
      {
        MITK_ERROR << "Error, dropping node " << i << ", from a list of " << droppedNodes.size() << " nodes into viewer " << dropIndex << ", could not find geometry which must be a programming bug." << std::endl;
        return;
      }

      // So we are removing all images that are present from the viewer denoted by dropIndex,
      if (!this->GetAccumulateWhenDropping())
      {
        this->RemoveNodesFromViewer(viewerDropInto);
      }

      // Initialise geometry according to first image
      if (!this->GetAccumulateWhenDropping())
      {
        viewerDropInto->SetTimeGeometry(timeGeometry.GetPointer());
        viewerDropInto->SetWindowLayout(windowLayout);
        viewerDropInto->SetEnabled(true);
      }

      // ...and then adding a single image to that viewer, denoted by dropIndex.
      this->AddNodeToViewer(viewerDropInto, droppedNodes[i]);

      // We need to always increment by at least one viewer, or else infinite loop-a-rama.
      dropIndex++;
    }
  }
  else if (m_DropType == DNDDISPLAY_DROP_ALL)
  {
    mitk::TimeGeometry::Pointer timeGeometry = this->GetTimeGeometry(droppedNodes, -1);
    if (timeGeometry.IsNull())
    {
      MITK_ERROR << "Error, dropping " << droppedNodes.size() << " nodes into viewer " << viewerIndex << ", could not find geometry which must be a programming bug." << std::endl;
      return;
    }

    // Clear all nodes from every viewer.
    if (!this->GetAccumulateWhenDropping())
    {
      this->ClearViewers();
    }

    // Note: Remember that we have window layout = axial, coronal, sagittal, 3D and ortho (+ others maybe)
    // So this thumbnail drop, has to switch to a single orientation. If the current default
    // window layout is not a single slice mode, we need to switch to one.
    WindowOrientation orientation = WINDOW_ORIENTATION_UNKNOWN;
    switch (windowLayout)
    {
    case WINDOW_LAYOUT_AXIAL:
      orientation = WINDOW_ORIENTATION_AXIAL;
      break;
    case WINDOW_LAYOUT_SAGITTAL:
      orientation = WINDOW_ORIENTATION_SAGITTAL;
      break;
    case WINDOW_LAYOUT_CORONAL:
      orientation = WINDOW_ORIENTATION_CORONAL;
      break;
    default:
      orientation = WINDOW_ORIENTATION_AXIAL;
      windowLayout = WINDOW_LAYOUT_AXIAL;
      break;
    }

    // Then we need to check if the number of slices < the number of viewers, if so, we just
    // spread the slices, one per viewer, until we run out of viewers.
    //
    // If we have more slices than viewers, we need to interpolate the number of slices.
    if (!this->GetAccumulateWhenDropping())
    {
      m_Viewers[0]->SetTimeGeometry(timeGeometry.GetPointer());
      m_Viewers[0]->SetWindowLayout(windowLayout);
    }

    int maxSlice = m_Viewers[0]->GetMaxSlice(orientation);
    int numberOfSlices = maxSlice + 1;
    std::size_t numberOfViewersToUse = std::min((std::size_t)numberOfSlices, (std::size_t)m_Viewers.size());

    // Now decide how we calculate which viewer is showing which slice.
    if (numberOfSlices <= m_Viewers.size())
    {
      // In this method, we have less slices than viewers, so we just spread them in increasing order.
      for (std::size_t i = 0; i < numberOfViewersToUse; i++)
      {
        if (!this->GetAccumulateWhenDropping())
        {
          m_Viewers[i]->SetTimeGeometry(timeGeometry.GetPointer());
          m_Viewers[i]->SetWindowLayout(windowLayout);
          m_Viewers[i]->SetEnabled(true);
        }
        m_Viewers[i]->SetSelectedSlice(orientation, i);
        m_Viewers[i]->FitToDisplay();
      }
    }
    else
    {
      // In this method, we have more slices than viewers, so we spread them evenly over the max number of viewers.
      for (std::size_t i = 0; i < numberOfViewersToUse; i++)
      {
        if (!this->GetAccumulateWhenDropping())
        {
          m_Viewers[i]->SetTimeGeometry(timeGeometry.GetPointer());
          m_Viewers[i]->SetWindowLayout(windowLayout);
          m_Viewers[i]->SetEnabled(true);
        }
        int numberOfEdgeSlicesToIgnore = static_cast<int>(numberOfSlices * 0.05); // ignore first and last 5 percent, as usually junk/blank.
        int remainingNumberOfSlices = numberOfSlices - (2 * numberOfEdgeSlicesToIgnore);
        float fraction = static_cast<float>(i) / m_Viewers.size();

        int chosenSlice = numberOfEdgeSlicesToIgnore + static_cast<int>(remainingNumberOfSlices * fraction);
        m_Viewers[i]->SetSelectedSlice(orientation, chosenSlice);
        m_Viewers[i]->FitToDisplay();
      }
    } // end if (which method of spreading thumbnails)

    // Now add the nodes to the right number of viewers.
    for (std::size_t i = 0; i < numberOfViewersToUse; i++)
    {
      for (std::size_t j = 0; j < droppedNodes.size(); j++)
      {
        this->AddNodeToViewer(m_Viewers[i], droppedNodes[j]);
      }
    }
  } // end if (which method of dropping)

  this->UpdateGlobalVisibilities(viewer->GetSelectedRenderWindow()->GetRenderer());
}

}
