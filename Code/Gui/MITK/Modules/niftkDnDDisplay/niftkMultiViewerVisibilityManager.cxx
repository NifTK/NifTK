/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMultiViewerVisibilityManager.h"
#include <QmitkRenderWindow.h>
#include "niftkSingleViewerWidget.h"
#include <mitkBaseRenderer.h>
#include <mitkVtkResliceInterpolationProperty.h>
#include <mitkDataStorageUtils.h>
#include <mitkImageAccessByItk.h>
#include <vtkRenderWindow.h>
#include <itkConversionUtils.h>
#include <itkSpatialOrientationAdapter.h>


//-----------------------------------------------------------------------------
niftkMultiViewerVisibilityManager::niftkMultiViewerVisibilityManager(mitk::DataStorage::Pointer dataStorage)
: m_BlockDataStorageEvents(false)
, m_AutomaticallyAddChildren(true)
, m_Accumulate(false)
{
  assert(dataStorage);
  m_DataStorage = dataStorage;

  m_DataNodes.clear();
  m_Viewers.clear();
  m_ObserverToVisibilityMap.clear();

  m_DataStorage->AddNodeEvent.AddListener(
      mitk::MessageDelegate1<niftkMultiViewerVisibilityManager, const mitk::DataNode*>
    ( this, &niftkMultiViewerVisibilityManager::NodeAddedProxy ) );

  m_DataStorage->RemoveNodeEvent.AddListener(
      mitk::MessageDelegate1<niftkMultiViewerVisibilityManager, const mitk::DataNode*>
    ( this, &niftkMultiViewerVisibilityManager::NodeRemovedProxy ) );
}


//-----------------------------------------------------------------------------
niftkMultiViewerVisibilityManager::~niftkMultiViewerVisibilityManager()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.RemoveListener(
        mitk::MessageDelegate1<niftkMultiViewerVisibilityManager, const mitk::DataNode*>
    ( this, &niftkMultiViewerVisibilityManager::NodeAddedProxy ));

    m_DataStorage->RemoveNodeEvent.RemoveListener(
        mitk::MessageDelegate1<niftkMultiViewerVisibilityManager, const mitk::DataNode*>
    ( this, &niftkMultiViewerVisibilityManager::NodeRemovedProxy ));

    m_DataStorage = NULL;
  }
  this->RemoveAllFromObserverToVisibilityMap();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::RemoveAllFromObserverToVisibilityMap()
{
  for( std::map<unsigned long, mitk::BaseProperty::Pointer>::iterator iter = m_ObserverToVisibilityMap.begin(); iter != m_ObserverToVisibilityMap.end(); ++iter )
  {
    (*iter).second->RemoveObserver((*iter).first);
  }
  m_ObserverToVisibilityMap.clear();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::UpdateObserverToVisibilityMap()
{
  this->RemoveAllFromObserverToVisibilityMap();

  assert(m_DataStorage);

  mitk::DataStorage::SetOfObjects::ConstPointer all = m_DataStorage->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    if (it->Value().IsNull() || it->Value()->GetProperty("visible") == NULL)
    {
      continue;
    }

    bool isHelper = false;
    it->Value()->GetBoolProperty("helper object", isHelper);

    if (isHelper)
    {
      continue;
    }

    /* register listener for changes in visible property */
    itk::ReceptorMemberCommand<niftkMultiViewerVisibilityManager>::Pointer command = itk::ReceptorMemberCommand<niftkMultiViewerVisibilityManager>::New();
    command->SetCallbackFunction(this, &niftkMultiViewerVisibilityManager::UpdateVisibilityProperty);
    m_ObserverToVisibilityMap[it->Value()->GetProperty("visible")->AddObserver( itk::ModifiedEvent(), command )] = it->Value()->GetProperty("visible");
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::RegisterViewer(niftkSingleViewerWidget *viewer)
{
  viewer->SetDataStorage(m_DataStorage);

  std::set<mitk::DataNode*> newNodes;
  m_DataNodes.push_back(newNodes);
  m_Viewers.push_back(viewer);

  this->SetAllNodeVisibilityForViewer(m_Viewers.size() - 1, false);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::DeregisterViewers(std::size_t startIndex, std::size_t endIndex)
{
  if (endIndex == -1)
  {
    endIndex = m_Viewers.size();
  }
  for (std::size_t i = startIndex; i < endIndex; ++i)
  {
    this->RemoveNodesFromViewer(i);
  }
  m_DataNodes.erase(m_DataNodes.begin() + startIndex, m_DataNodes.begin() + endIndex);
  m_Viewers.erase(m_Viewers.begin() + startIndex, m_Viewers.begin() + endIndex);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::ClearViewers(std::size_t startIndex, std::size_t endIndex)
{
  if (endIndex == -1)
  {
    endIndex = m_Viewers.size();
  }
  for (std::size_t i = startIndex; i < endIndex; i++)
  {
    this->RemoveNodesFromViewer(i);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::SetAllNodeVisibilityForAllViewers(bool visibility)
{
  assert(m_DataStorage);

  mitk::DataStorage::SetOfObjects::ConstPointer all = m_DataStorage->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    if (it->Value().IsNull() || it->Value()->GetProperty("visible") == NULL)
    {
      continue;
    }
    this->SetNodeVisibilityForAllViewers(it->Value(), visibility);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::SetNodeVisibilityForAllViewers(mitk::DataNode* node, bool visibility)
{
  for (std::size_t i = 0; i < m_Viewers.size(); i++)
  {
    this->SetNodeVisibilityForViewer(node, i, visibility);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::SetAllNodeVisibilityForViewer(std::size_t viewerIndex, bool visibility)
{
  assert(m_DataStorage);

  mitk::DataStorage::SetOfObjects::ConstPointer all = m_DataStorage->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    if (it->Value().IsNull() || it->Value()->GetProperty("visible") == NULL)
    {
      continue;
    }
    this->SetNodeVisibilityForViewer(it->Value(), viewerIndex, visibility);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::SetNodeVisibilityForViewer(mitk::DataNode* node, std::size_t viewerIndex, bool visibility)
{
  std::vector<mitk::DataNode*> nodes;
  nodes.push_back(node);
  m_Viewers[viewerIndex]->SetRendererSpecificVisibility(nodes, visibility);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::NodeRemovedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeRemoved()
  if(!m_BlockDataStorageEvents)
  {
    m_BlockDataStorageEvents = true;
    this->NodeRemoved(node);
    m_BlockDataStorageEvents = false;
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::NodeRemoved( const mitk::DataNode* node)
{
  this->UpdateObserverToVisibilityMap();

  for (std::size_t i = 0; i < m_DataNodes.size(); i++)
  {
    std::set<mitk::DataNode*>::iterator iter;
    iter = m_DataNodes[i].find(const_cast<mitk::DataNode*>(node));
    if (iter != m_DataNodes[i].end())
    {
      m_DataNodes[i].erase(iter);
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::NodeAddedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeAdded()
  if(!m_BlockDataStorageEvents)
  {
    m_BlockDataStorageEvents = true;
    this->NodeAdded(node);
    m_BlockDataStorageEvents = false;
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::NodeAdded(const mitk::DataNode* node)
{
  this->SetInitialNodeProperties(const_cast<mitk::DataNode*>(node));
  this->UpdateObserverToVisibilityMap();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::SetInitialNodeProperties(mitk::DataNode* node)
{
  // So as each new node is added (i.e. surfaces, point sets, images) we set default visibility to false.
  this->SetNodeVisibilityForAllViewers(node, false);

  bool globalVisibility = false;
  node->GetBoolProperty("visible", globalVisibility);

  // Furthermore, if a node has a parent, and that parent is already visible, we add this new node to all the same
  // viewer as its parent. This is useful in segmentation when we add a segmentation (binary) volume that is
  // registered as a child of a grey scale image. If the parent grey scale image is already
  // registered as visible in a viewer, then the child image is made visible, which has the effect of
  // immediately showing the segmented volume.
  mitk::DataNode::Pointer parent = mitk::FindParentGreyScaleImage(m_DataStorage, node);
  if (parent.IsNotNull())
  {
    for (std::size_t i = 0; i < m_DataNodes.size(); i++)
    {
      std::set<mitk::DataNode*>::iterator iter;
      for (iter = m_DataNodes[i].begin(); iter != m_DataNodes[i].end(); iter++)
      {
        if (*iter == parent)
        {
          this->AddNodeToViewer(i, node, globalVisibility);
        }
      }
    }
  }
  else
  {
    /// TODO This should not be handled here.
    if (node->GetName() == std::string("One of FeedbackContourTool's feedback nodes"))
    {
      for (std::size_t viewerIndex = 0; viewerIndex < m_Viewers.size(); ++viewerIndex)
      {
        if (m_Viewers[viewerIndex]->IsFocused())
        {
          this->AddNodeToViewer(viewerIndex, node, globalVisibility);
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::UpdateVisibilityProperty(const itk::EventObject&)
{
  // We have to iterate through all nodes registered with DataStorage,
  // and see if the global visibility property should override the renderer specific one for any
  // node that is registered as having been dropped into and of our registered render windows.

  // Outer loop, iterates through all nodes in DataStorage.
  mitk::DataStorage::SetOfObjects::ConstPointer all = m_DataStorage->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator dataStorageIterator = all->Begin(); dataStorageIterator != all->End(); ++dataStorageIterator)
  {
    // Make sure each node is non-NULL and has a non-NULL "visible" property and is not a helper object
    if (dataStorageIterator->Value().IsNull() || dataStorageIterator->Value()->GetProperty("visible") == NULL)
    {
      continue;
    }

    bool isHelper = false;
    dataStorageIterator->Value()->GetBoolProperty("helper object", isHelper);

    if (isHelper)
    {
      continue;
    }

    // Then we iterate through our list of viewers.
    for (std::size_t i = 0; i < m_DataNodes.size(); i++)
    {

      // And for each viewer, we have a set of registered nodes.
      std::set<mitk::DataNode*>::iterator nodesPerViewerIter;
      for (nodesPerViewerIter = m_DataNodes[i].begin(); nodesPerViewerIter != m_DataNodes[i].end(); nodesPerViewerIter++)
      {
        if (dataStorageIterator->Value() == (*nodesPerViewerIter))
        {
          bool globalVisibility(false);
          dataStorageIterator->Value()->GetBoolProperty("visible", globalVisibility);

          std::vector<mitk::DataNode*> nodes;
          nodes.push_back(dataStorageIterator->Value());

          m_Viewers[i]->SetRendererSpecificVisibility(nodes, globalVisibility);
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::RemoveNodesFromViewer(int viewerIndex)
{
  niftkSingleViewerWidget *viewer = m_Viewers[viewerIndex];
  assert(viewer);

  std::vector<mitk::DataNode*> nodes;
  std::set<mitk::DataNode*>::iterator iter;

  for (iter = m_DataNodes[viewerIndex].begin(); iter != m_DataNodes[viewerIndex].end(); iter++)
  {
    nodes.push_back(*iter);
  }

  viewer->SetRendererSpecificVisibility(nodes, false);
  m_DataNodes[viewerIndex].clear();
}


//-----------------------------------------------------------------------------
int niftkMultiViewerVisibilityManager::GetNodesInViewer(int viewerIndex)
{
  int result = m_DataNodes[viewerIndex].size();
  return result;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::AddNodeToViewer(int viewerIndex, mitk::DataNode* node, bool initialVisibility)
{
  niftkSingleViewerWidget* viewer = m_Viewers[viewerIndex];
  assert(viewer);

  m_DataNodes[viewerIndex].insert(node);
  node->Modified();

  std::vector<mitk::DataNode*> nodes;
  nodes.push_back(node);

  if (m_AutomaticallyAddChildren)
  {
    assert(m_DataStorage);

    mitk::DataStorage::SetOfObjects::Pointer possibleChildren = mitk::FindDerivedVisibleNonHelperChildren(m_DataStorage, node);
    for (std::size_t i = 0; i < possibleChildren->size(); i++)
    {
      mitk::DataNode* possibleNode = (*possibleChildren)[i];

      m_DataNodes[viewerIndex].insert(possibleNode);
      possibleNode->Modified();

      nodes.push_back(possibleNode);
    }
  }

  viewer->SetRendererSpecificVisibility(nodes, initialVisibility);

}


//-----------------------------------------------------------------------------
mitk::TimeGeometry::Pointer niftkMultiViewerVisibilityManager::GetGeometry(std::vector<mitk::DataNode*> nodes, int nodeIndex)
{
  mitk::TimeGeometry::Pointer geometry = NULL;
  int indexThatWeActuallyUsed = -1;

  // If nodeIndex < 0, we are choosing the best geometry from all available nodes.
  if (nodeIndex < 0)
  {

    // First try to find an image geometry, and if so, use the first one.
    mitk::Image::Pointer image = NULL;
    for (std::size_t i = 0; i < nodes.size(); i++)
    {
      image = dynamic_cast<mitk::Image*>(nodes[i]->GetData());
      if (image.IsNotNull())
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
        mitk::BaseData::Pointer data = nodes[i]->GetData();
        if (data.IsNotNull())
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
    mitk::BaseData::Pointer data = nodes[nodeIndex]->GetData();
    if (data.IsNotNull())
    {
      geometry = data->GetTimeGeometry();
      indexThatWeActuallyUsed = nodeIndex;
    }
  }
  // Essentially, the nodeIndex is garbage, so just pick the first one.
  else
  {
    mitk::BaseData::Pointer data = nodes[0]->GetData();
    if (data.IsNotNull())
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
    if (!mitk::IsNodeAGreyScaleImage(nodes[indexThatWeActuallyUsed]))
    {
      mitk::DataNode::Pointer node = FindParentGreyScaleImage(m_DataStorage, nodes[indexThatWeActuallyUsed]);
      if (node.IsNotNull())
      {
        mitk::BaseData::Pointer data = nodes[0]->GetData();
        if (data.IsNotNull())
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
niftkMultiViewerVisibilityManager::GetAsAcquiredOrientation(
    itk::Image<TPixel, VImageDimension>* itkImage,
    MIDASOrientation &outputOrientation
    )
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
      outputOrientation = MIDAS_ORIENTATION_AXIAL;
    }
    else
    {
      outputOrientation = MIDAS_ORIENTATION_CORONAL;
    }
  }
  else if (orientationString[0] == 'A' || orientationString[0] == 'P')
  {
    if (orientationString[1] == 'L' || orientationString[1] == 'R')
    {
      outputOrientation = MIDAS_ORIENTATION_AXIAL;
    }
    else
    {
      outputOrientation = MIDAS_ORIENTATION_SAGITTAL;
    }
  }
  else if (orientationString[0] == 'S' || orientationString[0] == 'I')
  {
    if (orientationString[1] == 'L' || orientationString[1] == 'R')
    {
      outputOrientation = MIDAS_ORIENTATION_CORONAL;
    }
    else
    {
      outputOrientation = MIDAS_ORIENTATION_SAGITTAL;
    }
  }
}


//-----------------------------------------------------------------------------
WindowLayout niftkMultiViewerVisibilityManager::GetWindowLayout(std::vector<mitk::DataNode*> nodes)
{

  WindowLayout windowLayout = m_DefaultWindowLayout;
  if (windowLayout == WINDOW_LAYOUT_AS_ACQUIRED)
  {
    // "As Acquired" means you take the orientation of the XY plane
    // in the original image data, so we switch to ITK to work it out.
    MIDASOrientation orientation = MIDAS_ORIENTATION_CORONAL;

    mitk::Image::Pointer image = NULL;
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
        MITK_ERROR << "niftkMultiViewerVisibilityManager::OnNodesDropped failed to work out 'As Acquired' orientation." << e.what() << std::endl;
      }
    }
    else
    {
      MITK_ERROR << "niftkMultiViewerVisibilityManager::OnNodesDropped failed to find an image to work out 'As Acquired' orientation." << std::endl;
    }

    if (orientation == MIDAS_ORIENTATION_AXIAL)
    {
      windowLayout = WINDOW_LAYOUT_AXIAL;
    }
    else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
    {
      windowLayout = WINDOW_LAYOUT_SAGITTAL;
    }
    else if (orientation == MIDAS_ORIENTATION_CORONAL)
    {
      windowLayout = WINDOW_LAYOUT_CORONAL;
    }
    else
    {
      MITK_ERROR << "niftkMultiViewerVisibilityManager::OnNodesDropped defaulting to window layout " << windowLayout << std::endl;
    }
  }
  return windowLayout;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerVisibilityManager::OnNodesDropped(niftkSingleViewerWidget* viewer, QmitkRenderWindow *renderWindow, std::vector<mitk::DataNode*> nodes)
{
  int viewerIndex = std::find(m_Viewers.begin(), m_Viewers.end(), viewer) - m_Viewers.begin();
  WindowLayout windowLayout = this->GetWindowLayout(nodes);

  if (m_DataStorage.IsNotNull() && viewerIndex != -1)
  {
    for (std::size_t i = 0; i < nodes.size(); i++)
    {
      std::string name;
      if (nodes[i] != 0 && nodes[i]->GetStringProperty("name", name))
      {
        MITK_DEBUG << "Dropped " << nodes.size() << " into viewer[" << viewerIndex <<"], name[" << i << "]=" << name << std::endl;
      }
    }

    if (m_DropType == DNDDISPLAY_DROP_SINGLE)
    {

      MITK_DEBUG << "Dropped single" << std::endl;

      mitk::TimeGeometry::Pointer geometry = this->GetGeometry(nodes, -1);
      if (geometry.IsNull())
      {
        MITK_ERROR << "Error, dropping " << nodes.size() << " nodes into viewer " << viewerIndex << ", could not find geometry which must be a programming bug." << std::endl;
        return;
      }

      // Clear all nodes from the single viewer denoted by viewerIndex (the one that was dropped into).
      if (this->GetNodesInViewer(viewerIndex) > 0 && !this->GetAccumulateWhenDropped())
      {
        this->RemoveNodesFromViewer(viewerIndex);
      }

      // Then set up geometry of that single viewer.
      if (this->GetNodesInViewer(viewerIndex) == 0 || !this->GetAccumulateWhenDropped())
      {
        m_Viewers[viewerIndex]->SetGeometry(geometry.GetPointer());
        m_Viewers[viewerIndex]->SetWindowLayout(windowLayout);
        m_Viewers[viewerIndex]->SetEnabled(true);
      }

      // Then add all nodes into the same viewer denoted by viewerIndex (the one that was dropped into).
      for (std::size_t i = 0; i < nodes.size(); i++)
      {
        this->AddNodeToViewer(viewerIndex, nodes[i]);
      }
    }
    else if (m_DropType == DNDDISPLAY_DROP_MULTIPLE)
    {
      MITK_DEBUG << "Dropped multiple" << std::endl;

      // Work out which viewer we are actually dropping into.
      // We aim to put one object, in each of consecutive viewers.
      // If we hit the end (of the 5x5=25 list), we go back to zero.

      std::size_t dropIndex = viewerIndex;

      for (std::size_t i = 0; i < nodes.size(); i++)
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

        mitk::TimeGeometry::Pointer geometry = this->GetGeometry(nodes, i);
        if (geometry.IsNull())
        {
          MITK_ERROR << "Error, dropping node " << i << ", from a list of " << nodes.size() << " nodes into viewer " << dropIndex << ", could not find geometry which must be a programming bug." << std::endl;
          return;
        }

        // So we are removing all images that are present from the viewer denoted by dropIndex,
        if (this->GetNodesInViewer(dropIndex) > 0 && !this->GetAccumulateWhenDropped())
        {
          this->RemoveNodesFromViewer(dropIndex);
        }

        // Initialise geometry according to first image
        if (this->GetNodesInViewer(dropIndex) == 0 || !this->GetAccumulateWhenDropped())
        {
          m_Viewers[dropIndex]->SetGeometry(geometry.GetPointer());
          m_Viewers[dropIndex]->SetWindowLayout(windowLayout);
          m_Viewers[dropIndex]->SetEnabled(true);
        }

        // ...and then adding a single image to that viewer, denoted by dropIndex.
        this->AddNodeToViewer(dropIndex, nodes[i]);

        // We need to always increment by at least one viewer, or else infinite loop-a-rama.
        dropIndex++;
      }
    }
    else if (m_DropType == DNDDISPLAY_DROP_ALL)
    {
      MITK_DEBUG << "Dropped thumbnail" << std::endl;

      mitk::TimeGeometry::Pointer geometry = this->GetGeometry(nodes, -1);
      if (geometry.IsNull())
      {
        MITK_ERROR << "Error, dropping " << nodes.size() << " nodes into viewer " << viewerIndex << ", could not find geometry which must be a programming bug." << std::endl;
        return;
      }

      // Clear all nodes from every viewer.
      if (this->GetNodesInViewer(0) > 0 && !this->GetAccumulateWhenDropped())
      {
        this->ClearViewers();
      }

      // Note: Remember that we have window layout = axial, coronal, sagittal, 3D and ortho (+ others maybe)
      // So this thumbnail drop, has to switch to a single orientation. If the current default
      // window layout is not a single slice mode, we need to switch to one.
      MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;
      switch (windowLayout)
      {
      case WINDOW_LAYOUT_AXIAL:
        orientation = MIDAS_ORIENTATION_AXIAL;
        break;
      case WINDOW_LAYOUT_SAGITTAL:
        orientation = MIDAS_ORIENTATION_SAGITTAL;
        break;
      case WINDOW_LAYOUT_CORONAL:
        orientation = MIDAS_ORIENTATION_CORONAL;
        break;
      default:
        orientation = MIDAS_ORIENTATION_AXIAL;
        windowLayout = WINDOW_LAYOUT_AXIAL;
        break;
      }

      // Then we need to check if the number of slices < the number of viewers, if so, we just
      // spread the slices, one per viewer, until we run out of viewers.
      //
      // If we have more slices than viewers, we need to interpolate the number of slices.
      if (this->GetNodesInViewer(viewerIndex) == 0 || !this->GetAccumulateWhenDropped())
      {
        m_Viewers[0]->SetGeometry(geometry.GetPointer());
        m_Viewers[0]->SetWindowLayout(windowLayout);
      }

      int maxSlice = m_Viewers[0]->GetMaxSlice(orientation);
      int numberOfSlices = maxSlice + 1;
      std::size_t viewersToUse = std::min((std::size_t)numberOfSlices, (std::size_t)m_Viewers.size());

      MITK_DEBUG << "Dropping thumbnail, maxSlice=" << maxSlice << ", numberOfSlices=" << numberOfSlices << ", viewersToUse=" << viewersToUse << std::endl;

      // Now decide how we calculate which viewer is showing which slice.
      if (numberOfSlices <= m_Viewers.size())
      {
        // In this method, we have less slices than viewers, so we just spread them in increasing order.
        for (std::size_t i = 0; i < viewersToUse; i++)
        {
          if (this->GetNodesInViewer(i) == 0 || !this->GetAccumulateWhenDropped())
          {
            m_Viewers[i]->SetGeometry(geometry.GetPointer());
            m_Viewers[i]->SetWindowLayout(windowLayout);
            m_Viewers[i]->SetEnabled(true);
          }
          m_Viewers[i]->SetSelectedSlice(orientation, i);
          m_Viewers[i]->FitToDisplay();
          MITK_DEBUG << "Dropping thumbnail, slice=" << i << std::endl;
        }
      }
      else
      {
        // In this method, we have more slices than viewers, so we spread them evenly over the max number of viewers.
        for (std::size_t i = 0; i < viewersToUse; i++)
        {
          if (this->GetNodesInViewer(i) == 0 || !this->GetAccumulateWhenDropped())
          {
            m_Viewers[i]->SetGeometry(geometry.GetPointer());
            m_Viewers[i]->SetWindowLayout(windowLayout);
            m_Viewers[i]->SetEnabled(true);
          }
          int maxSlice = m_Viewers[i]->GetMaxSlice(orientation);
          int numberOfEdgeSlicesToIgnore = static_cast<int>(numberOfSlices * 0.05); // ignore first and last 5 percent, as usually junk/blank.
          int remainingNumberOfSlices = numberOfSlices - (2 * numberOfEdgeSlicesToIgnore);
          float fraction = static_cast<float>(i) / m_Viewers.size();

          int chosenSlice = numberOfEdgeSlicesToIgnore + static_cast<int>(remainingNumberOfSlices * fraction);

          MITK_DEBUG << "Dropping thumbnail, i=" << i \
              << ", maxSlice=" << maxSlice \
              << ", numberOfEdgeSlicesToIgnore=" << numberOfEdgeSlicesToIgnore \
              << ", remainingNumberOfSlices=" << remainingNumberOfSlices \
              << ", fraction=" << fraction \
              << ", chosenSlice=" << chosenSlice << std::endl;
          m_Viewers[i]->SetSelectedSlice(orientation, chosenSlice);
          m_Viewers[i]->FitToDisplay();
        }
      } // end if (which method of spreading thumbnails)

      // Now add the nodes to the right number of viewers.
      for (std::size_t i = 0; i < viewersToUse; i++)
      {
        for (std::size_t j = 0; j < nodes.size(); j++)
        {
          this->AddNodeToViewer(i, nodes[j]);
        }
      }
    } // end if (which method of dropping)
  } // end if (we have valid input)
}
