/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkDataStorageVisibilityTracker.h"
#include "mitkBaseRenderer.h"

namespace mitk
{

//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::Init(const mitk::DataStorage::Pointer dataStorage)
{
  m_Listener = mitk::DataStoragePropertyListener::New();
  m_Listener->SetPropertyName("visible");
  m_Listener->SetDataStorage(dataStorage);
  m_Listener->SetAutoFire(true);

  m_RenderersToTrack.clear();
  m_RenderersToUpdate.clear();
  m_ExcludedNodeList.clear();

  m_Listener->PropertyChanged += mitk::MessageDelegate<DataStorageVisibilityTracker>( this, &DataStorageVisibilityTracker::OnPropertyChanged );
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
  // m_Listener destroyed via smart pointer.
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::SetRenderersToUpdate(std::vector<mitk::BaseRenderer*>& list)
{
  m_RenderersToUpdate = list;
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStorageVisibilityTracker::SetRenderersToTrack(std::vector<mitk::BaseRenderer*>& list)
{
  m_RenderersToTrack = list;
  m_Listener->SetRenderers(list);
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
void DataStorageVisibilityTracker::SetNodesToIgnore(std::vector<mitk::DataNode*>& nodes)
{
  m_ExcludedNodeList = nodes;
}


//-----------------------------------------------------------------------------
bool DataStorageVisibilityTracker::IsExcluded(mitk::DataNode* node)
{
  bool isExcluded = false;

  std::vector<mitk::DataNode*>::iterator iter;
  for (iter = m_ExcludedNodeList.begin(); iter != m_ExcludedNodeList.end(); iter++)
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
void DataStorageVisibilityTracker::OnPropertyChanged()
{
  if (m_DataStorage.IsNotNull())
  {
    // block the calls, so we can update stuff, without repeated callback loops.
    m_Listener->SetBlock(true);

    // Intention : This object should display all the data nodes visible in the focused window, and none others.
    // Assumption: Renderer specific properties override the global ones.
    // so......    Objects will be visible, unless the the node has a render window specific property that says otherwise.

    if (m_RenderersToTrack.size() > 0 && m_RenderersToUpdate.size() > 0)
    {
      mitk::DataStorage::SetOfObjects::ConstPointer allNodes = m_DataStorage->GetAll();
      mitk::DataStorage::SetOfObjects::const_iterator allNodesIter;

      for (allNodesIter = allNodes->begin(); allNodesIter != allNodes->end(); ++allNodesIter)
      {
        if (!this->IsExcluded(*allNodesIter))
        {
          bool globalVisible(false);
          bool foundGlobalVisible(false);
          foundGlobalVisible = (*allNodesIter)->GetBoolProperty("visible", globalVisible);

          for (unsigned int i = 0; i < m_RenderersToTrack.size(); i++)
          {

            bool trackedWindowVisible(false);
            bool foundTrackedWindowVisible(false);
            foundTrackedWindowVisible = (*allNodesIter)->GetBoolProperty("visible", trackedWindowVisible, m_RenderersToTrack[i]);

            // We default to ON.
            bool finalVisibility(true);

            // The logic.
            if ((foundTrackedWindowVisible && !trackedWindowVisible)
                || (foundGlobalVisible && !globalVisible)
                )
            {
              finalVisibility = false;
            }

            // Set the final visibility flag
            for (unsigned int j = 0; j < m_RenderersToUpdate.size(); j++)
            {
              (*allNodesIter)->SetBoolProperty("visible", finalVisibility, m_RenderersToUpdate[j]);
            }

          } // end for i
        } // end if not excluded
      } // end for each node
    } // end if we did actually have some renderers to track

    // don't forget to unblock.
    m_Listener->SetBlock(false);

  } // end if data storage
}

} // end namespace
