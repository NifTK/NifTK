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
DataStorageListener::DataStorageListener()
: m_DataStorage(NULL)
, m_InDataStorageChanged(false)
, m_Block(false)
{
  m_Filters.clear();
}


//-----------------------------------------------------------------------------
DataStorageListener::DataStorageListener(const mitk::DataStorage::Pointer dataStorage)
: m_DataStorage(dataStorage)
, m_InDataStorageChanged(false)
{
  this->Activate(dataStorage);
}


//-----------------------------------------------------------------------------
DataStorageListener::~DataStorageListener()
{
  this->Deactivate();
}


//-----------------------------------------------------------------------------
void DataStorageListener::SetDataStorage(const mitk::DataStorage::Pointer dataStorage)
{
  this->Activate(dataStorage);
}


//-----------------------------------------------------------------------------
void DataStorageListener::AddFilter(mitk::DataNodeFilter::Pointer filter)
{
  m_Filters.push_back(filter);
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStorageListener::ClearFilters()
{
  m_Filters.clear();
  this->Modified();
}


//-----------------------------------------------------------------------------
bool DataStorageListener::Pass(const mitk::DataNode* node)
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
void DataStorageListener::Activate(const mitk::DataStorage::Pointer dataStorage)
{
  if (m_DataStorage.IsNotNull())
  {
    this->Deactivate();
  }

  if (dataStorage.IsNotNull())
  {
    m_DataStorage = dataStorage;

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

    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void DataStorageListener::Deactivate()
{
  if (m_DataStorage.IsNotNull())
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

    m_DataStorage = NULL;

    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void DataStorageListener::NodeAddedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeAdded()
  if(!m_Block && m_DataStorage.IsNotNull() && node != NULL && !m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    if (this->Pass(node))
    {
      this->NodeAdded(const_cast<mitk::DataNode*>(node));
    }
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void DataStorageListener::NodeChangedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeRemoved()
  if(!m_Block && m_DataStorage.IsNotNull() && node != NULL && !m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    if (this->Pass(node))
    {
      this->NodeChanged(const_cast<mitk::DataNode*>(node));
    }
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void DataStorageListener::NodeRemovedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeRemoved()
  if(!m_Block && m_DataStorage.IsNotNull() && node != NULL && !m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    if (this->Pass(node))
    {
      this->NodeRemoved(const_cast<mitk::DataNode*>(node));
    }
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void DataStorageListener::NodeDeletedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeRemoved()
  if(!m_Block && m_DataStorage.IsNotNull() && node != NULL && !m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    if (this->Pass(node))
    {
      this->NodeDeleted(const_cast<mitk::DataNode*>(node));
    }
    m_InDataStorageChanged = false;
  }
}

} // end namespace
