/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleViewerWidgetListDropManager.h"
#include <mitkDataStorageUtils.h>
#include <mitkMIDASEnums.h>
#include <mitkMIDASImageUtils.h>
#include <mitkDataStorageUtils.h>
#include "niftkSingleViewerWidget.h"
#include "niftkSingleViewerWidgetListVisibilityManager.h"

//-----------------------------------------------------------------------------
niftkSingleViewerWidgetListDropManager::niftkSingleViewerWidgetListDropManager()
: m_DefaultLayout(MIDAS_LAYOUT_CORONAL)
, m_DropType(MIDAS_DROP_TYPE_SINGLE)
, m_DataStorage(NULL)
, m_VisibilityManager(NULL)
{
}


//-----------------------------------------------------------------------------
niftkSingleViewerWidgetListDropManager::~niftkSingleViewerWidgetListDropManager()
{

}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidgetListDropManager::SetVisibilityManager(niftkSingleViewerWidgetListVisibilityManager* manager)
{
  m_VisibilityManager = manager;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidgetListDropManager::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  m_DataStorage = dataStorage;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidgetListDropManager::SetDefaultLayout(MIDASLayout layout)
{
  m_DefaultLayout = layout;
}


//-----------------------------------------------------------------------------
MIDASLayout niftkSingleViewerWidgetListDropManager::GetDefaultLayout() const
{
  return m_DefaultLayout;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidgetListDropManager::SetDropType(const MIDASDropType& dropType)
{
  m_DropType = dropType;
}


//-----------------------------------------------------------------------------
MIDASDropType niftkSingleViewerWidgetListDropManager::GetDropType() const
{
  return m_DropType;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidgetListDropManager::SetAccumulateWhenDropped(const bool& accumulateWhenDropped)
{
  m_AccumulateWhenDropped = accumulateWhenDropped;
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerWidgetListDropManager::GetAccumulateWhenDropped() const
{
  return m_AccumulateWhenDropped;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerWidgetListDropManager::OnNodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes)
{
  if (m_DataStorage.IsNull() || m_VisibilityManager == NULL)
  {
    return;
  }

  if (nodes.size() == 0)
  {
    MITK_ERROR << "Calling niftkSingleViewerWidgetListDropManager::OnNodesDropped with no nodes. Surely a programming bug?" << std::endl;
    return;
  }

  int windowIndex = this->GetIndexFromWindow(window);
  if (windowIndex < 0)
  {
    MITK_ERROR << "Calling niftkSingleViewerWidgetListDropManager::OnNodesDropped with an invalid window. Surely a bug?" << std::endl;
  }

  MIDASOrientation defaultOrientation = MIDAS_ORIENTATION_CORONAL;
  MIDASOrientation orientation = GetAsAcquiredOrientation(defaultOrientation, dynamic_cast<mitk::Image*>(nodes[0]->GetData()));
  MIDASLayout layout;
  if (orientation == MIDAS_ORIENTATION_AXIAL)
  {
    layout = MIDAS_LAYOUT_AXIAL;
  }
  else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    layout = MIDAS_LAYOUT_SAGITTAL;
  }
  else if (orientation == MIDAS_ORIENTATION_CORONAL)
  {
    layout = MIDAS_LAYOUT_CORONAL;
  }
  else
  {
    layout = MIDAS_LAYOUT_CORONAL;
  }

  for (unsigned int i = 0; i < nodes.size(); i++)
  {
    std::string name;
    if (nodes[i] != 0 && nodes[i]->GetStringProperty("name", name))
    {
      MITK_DEBUG << "Dropped " << nodes.size() << " into window[" << windowIndex <<"], name[" << i << "]=" << name << std::endl;
    }
  }

  if (m_DropType == MIDAS_DROP_TYPE_SINGLE)
  {

    MITK_DEBUG << "Dropped single" << std::endl;

    mitk::TimeGeometry::Pointer geometry = mitk::GetPreferredGeometry(m_DataStorage, nodes, -1);
    if (geometry.IsNull())
    {
      MITK_ERROR << "Error, dropping " << nodes.size() << " nodes into window " << windowIndex << ", could not find geometry which must be a programming bug." << std::endl;
      return;
    }

    // Clear all nodes from the single window denoted by windowIndex (the one that was dropped into).
    if (this->GetNumberOfNodesRegisteredWithWidget(windowIndex) > 0 && !this->GetAccumulateWhenDropped())
    {
      m_VisibilityManager->ClearWindow(windowIndex);
    }

    // Then set up geometry of that single window.
    if (this->GetNumberOfNodesRegisteredWithWidget(windowIndex) == 0 || !this->GetAccumulateWhenDropped())
    {
      m_Widgets[windowIndex]->SetGeometry(geometry.GetPointer());
      m_Widgets[windowIndex]->SetLayout(layout);
      m_Widgets[windowIndex]->SetEnabled(true);
    }

    // Then add all nodes into the same window denoted by windowIndex (the one that was dropped into).
    for (unsigned int i = 0; i < nodes.size(); i++)
    {
      m_VisibilityManager->SetNodeVisibilityForWindow(nodes[i], windowIndex, true);
    }
  }
  else if (m_DropType == MIDAS_DROP_TYPE_MULTIPLE)
  {
    MITK_DEBUG << "Dropped multiple" << std::endl;

    // Work out which window we are actually dropping into.
    // We aim to put one object, in each of consecutive windows.
    // If we hit the end (of the 5x5=25 list), we go back to zero.

    unsigned int dropIndex = windowIndex;

    for (unsigned int i = 0; i < nodes.size(); i++)
    {
      while (dropIndex < m_Widgets.size() && !m_Widgets[dropIndex]->isVisible())
      {
        // i.e. if the window we are in, is not visible, keep looking
        dropIndex++;
      }
      if (dropIndex == m_Widgets.size())
      {
        // give up? Or we could go back to zero?
        dropIndex = 0;
      }

      mitk::TimeGeometry::Pointer geometry = mitk::GetPreferredGeometry(m_DataStorage, nodes, i);
      if (geometry.IsNull())
      {
        MITK_ERROR << "Error, dropping node " << i << ", from a list of " << nodes.size() << " nodes into window " << dropIndex << ", could not find geometry which must be a programming bug." << std::endl;
        return;
      }

      // So we are removing all images that are present from the window denoted by dropIndex,
      if (this->GetNumberOfNodesRegisteredWithWidget(dropIndex) > 0 && !this->GetAccumulateWhenDropped())
      {
        m_VisibilityManager->ClearWindow(dropIndex);
      }

      // Initialise geometry according to first image
      if (this->GetNumberOfNodesRegisteredWithWidget(dropIndex) == 0 || !this->GetAccumulateWhenDropped())
      {
        m_Widgets[dropIndex]->SetGeometry(geometry.GetPointer());
        m_Widgets[dropIndex]->SetLayout(layout);
        m_Widgets[dropIndex]->SetEnabled(true);
      }

      // ...and then adding a single image to that window, denoted by dropIndex.
      m_VisibilityManager->SetNodeVisibilityForWindow(nodes[i], dropIndex, true);

      // We need to always increment by at least one window, or else infinite loop-a-rama.
      dropIndex++;
    }
  }
  else if (m_DropType == MIDAS_DROP_TYPE_ALL)
  {
    MITK_DEBUG << "Dropped thumbnail" << std::endl;

    mitk::TimeGeometry::Pointer geometry = mitk::GetPreferredGeometry(m_DataStorage, nodes, -1);
    if (geometry.IsNull())
    {
      MITK_ERROR << "Error, dropping " << nodes.size() << " nodes into window " << windowIndex << ", could not find geometry which must be a programming bug." << std::endl;
      return;
    }

    // Clear all nodes from every window.
    if (this->GetNumberOfNodesRegisteredWithWidget(0) > 0 && !this->GetAccumulateWhenDropped())
    {
      m_VisibilityManager->ClearAllWindows();
    }

    // Note: Remember that we have layout = axial, coronal, sagittal, 3D and ortho (+ others maybe)
    // So this thumbnail drop, has to switch to a single orientation. If the current default
    // layout is not a single slice mode, we need to switch to one.
    MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;
    switch (layout)
    {
    case MIDAS_LAYOUT_AXIAL:
      orientation = MIDAS_ORIENTATION_AXIAL;
      break;
    case MIDAS_LAYOUT_SAGITTAL:
      orientation = MIDAS_ORIENTATION_SAGITTAL;
      break;
    case MIDAS_LAYOUT_CORONAL:
      orientation = MIDAS_ORIENTATION_CORONAL;
      break;
    default:
      orientation = MIDAS_ORIENTATION_AXIAL;
      layout = MIDAS_LAYOUT_AXIAL;
      break;
    }

    // Then we need to check if the number of slices < the number of windows, if so, we just
    // spread the slices, one per window, until we run out of windows.
    //
    // If we have more slices than windows, we need to interpolate the number of slices.
    if (this->GetNumberOfNodesRegisteredWithWidget(windowIndex) == 0 || !this->GetAccumulateWhenDropped())
    {
      m_Widgets[0]->SetGeometry(geometry.GetPointer());
      m_Widgets[0]->SetLayout(layout);
    }

    unsigned int maxSliceIndex = m_Widgets[0]->GetMaxSliceIndex(orientation);
    unsigned int numberOfSlices = maxSliceIndex + 1;
    unsigned int windowsToUse = std::min((unsigned int)numberOfSlices, (unsigned int)m_Widgets.size());

    MITK_DEBUG << "Dropping thumbnail, maxSlice=" << maxSliceIndex << ", numberOfSlices=" << numberOfSlices << ", windowsToUse=" << windowsToUse << std::endl;

    // Now decide how we calculate which window is showing which slice.
    if (numberOfSlices <= m_Widgets.size())
    {
      // In this method, we have less slices than windows, so we just spread them in increasing order.
      for (unsigned int i = 0; i < windowsToUse; i++)
      {
        if (this->GetNumberOfNodesRegisteredWithWidget(i) == 0 || !this->GetAccumulateWhenDropped())
        {
          m_Widgets[i]->SetGeometry(geometry.GetPointer());
          m_Widgets[i]->SetLayout(layout);
          m_Widgets[i]->SetEnabled(true);
        }
        m_Widgets[i]->SetSliceIndex(orientation, i);
        m_Widgets[i]->FitToDisplay();
        MITK_DEBUG << "Dropping thumbnail, sliceIndex=" << i << std::endl;
      }
    }
    else
    {
      // In this method, we have more slices than windows, so we spread them evenly over the max number of windows.
      for (unsigned int i = 0; i < windowsToUse; i++)
      {
        if (this->GetNumberOfNodesRegisteredWithWidget(i) == 0 || !this->GetAccumulateWhenDropped())
        {
          m_Widgets[i]->SetGeometry(geometry.GetPointer());
          m_Widgets[i]->SetLayout(layout);
          m_Widgets[i]->SetEnabled(true);
        }
        unsigned int maxSliceIndex = m_Widgets[i]->GetMaxSliceIndex(orientation);
        unsigned int numberOfEdgeSlicesToIgnore = static_cast<unsigned int>(numberOfSlices * 0.05); // ignore first and last 5 percent, as usually junk/blank.
        unsigned int remainingNumberOfSlices = numberOfSlices - (2 * numberOfEdgeSlicesToIgnore);
        float fraction = static_cast<float>(i) / m_Widgets.size();
        unsigned int chosenSlice = numberOfEdgeSlicesToIgnore + static_cast<unsigned int>(remainingNumberOfSlices * fraction);

        MITK_DEBUG << "Dropping thumbnail, i=" << i \
            << ", maxSlice=" << maxSliceIndex \
            << ", numberOfEdgeSlicesToIgnore=" << numberOfEdgeSlicesToIgnore \
            << ", remainingNumberOfSlices=" << remainingNumberOfSlices \
            << ", fraction=" << fraction \
            << ", chosenSlice=" << chosenSlice << std::endl;
        m_Widgets[i]->SetSliceIndex(orientation, chosenSlice);
        m_Widgets[i]->FitToDisplay();
      }
    } // end if (which method of spreading thumbnails)

    // Now add the nodes to the right number of Windows.
    for (unsigned int i = 0; i < windowsToUse; i++)
    {
      for (unsigned int j = 0; j < nodes.size(); j++)
      {
        m_VisibilityManager->SetNodeVisibilityForWindow(nodes[i], i, true);
      }
    }
  } // end if (which method of dropping)
}
