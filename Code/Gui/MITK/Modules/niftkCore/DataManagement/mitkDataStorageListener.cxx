/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDataStorageListener.h"

namespace mitk
{

//-----------------------------------------------------------------------------
DataStorageListener::DataStorageListener(const mitk::DataStorage::Pointer dataStorage)
: m_DataStorage(dataStorage)
, m_InDataStorageChanged(false)
, m_Blocked(false)
{
  assert(m_DataStorage.IsNotNull());

  this->AddListeners();
}


//-----------------------------------------------------------------------------
DataStorageListener::~DataStorageListener()
{
  this->RemoveListeners();
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer DataStorageListener::GetDataStorage() const
{
  return m_DataStorage;
}


//-----------------------------------------------------------------------------
void DataStorageListener::AddFilter(mitk::DataNodeFilter::Pointer filter)
{
  m_Filters.push_back(filter);
}


//-----------------------------------------------------------------------------
void DataStorageListener::ClearFilters()
{
  m_Filters.clear();
}


//-----------------------------------------------------------------------------
bool DataStorageListener::IsBlocked() const
{
  return m_Blocked;
}


//-----------------------------------------------------------------------------
bool DataStorageListener::SetBlocked(bool blocked)
{
  bool wasBlocked = m_Blocked;
  m_Blocked = blocked;
  return wasBlocked;
}


//-----------------------------------------------------------------------------
bool DataStorageListener::Pass(const mitk::DataNode* node) const
{
  bool result = true;
  for (unsigned int i = 0; i < m_Filters.size(); i++)
  {
    if (!m_Filters[i]->Pass(node))
    {
      result = false;
      break;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
void DataStorageListener::AddListeners()
{
  m_DataStorage->AddNodeEvent.AddListener(
      mitk::MessageDelegate1<DataStorageListener, const mitk::DataNode*>
    ( this, &DataStorageListener::NodeAddedProxy ) );

  m_DataStorage->ChangedNodeEvent.AddListener(
      mitk::MessageDelegate1<DataStorageListener, const mitk::DataNode*>
    ( this, &DataStorageListener::NodeChangedProxy ) );

  m_DataStorage->RemoveNodeEvent.AddListener(
      mitk::MessageDelegate1<DataStorageListener, const mitk::DataNode*>
    ( this, &DataStorageListener::NodeRemovedProxy ) );

  m_DataStorage->DeleteNodeEvent.AddListener(
      mitk::MessageDelegate1<DataStorageListener, const mitk::DataNode*>
    ( this, &DataStorageListener::NodeDeletedProxy ) );
}


//-----------------------------------------------------------------------------
void DataStorageListener::RemoveListeners()
{
  m_DataStorage->AddNodeEvent.RemoveListener(
      mitk::MessageDelegate1<DataStorageListener, const mitk::DataNode*>
  ( this, &DataStorageListener::NodeAddedProxy ));

  m_DataStorage->ChangedNodeEvent.RemoveListener(
      mitk::MessageDelegate1<DataStorageListener, const mitk::DataNode*>
  ( this, &DataStorageListener::NodeChangedProxy ));

  m_DataStorage->RemoveNodeEvent.RemoveListener(
      mitk::MessageDelegate1<DataStorageListener, const mitk::DataNode*>
  ( this, &DataStorageListener::NodeRemovedProxy ));

  m_DataStorage->DeleteNodeEvent.RemoveListener(
      mitk::MessageDelegate1<DataStorageListener, const mitk::DataNode*>
  ( this, &DataStorageListener::NodeDeletedProxy ));
}


//-----------------------------------------------------------------------------
void DataStorageListener::NodeAddedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeAdded()
  if (!m_Blocked && node != NULL && !m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    if (this->Pass(node))
    {
      this->OnNodeAdded(const_cast<mitk::DataNode*>(node));
    }
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void DataStorageListener::NodeChangedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeRemoved()
  if (!m_Blocked && node != NULL && !m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    if (this->Pass(node))
    {
      this->OnNodeChanged(const_cast<mitk::DataNode*>(node));
    }
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void DataStorageListener::NodeRemovedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeRemoved()
  if (!m_Blocked && node != NULL && !m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    if (this->Pass(node))
    {
      this->OnNodeRemoved(const_cast<mitk::DataNode*>(node));
    }
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void DataStorageListener::NodeDeletedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeRemoved()
  if (!m_Blocked && node != NULL && !m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    if (this->Pass(node))
    {
      this->OnNodeDeleted(const_cast<mitk::DataNode*>(node));
    }
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void DataStorageListener::OnNodeAdded(mitk::DataNode* /*node*/)
{
}


//-----------------------------------------------------------------------------
void DataStorageListener::OnNodeChanged(mitk::DataNode* /*node*/)
{
}


//-----------------------------------------------------------------------------
void DataStorageListener::OnNodeRemoved(mitk::DataNode* /*node*/)
{
}


//-----------------------------------------------------------------------------
void DataStorageListener::OnNodeDeleted(mitk::DataNode* /*node*/)
{
}

}
