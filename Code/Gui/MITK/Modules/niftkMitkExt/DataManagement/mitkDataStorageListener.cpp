/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkDataStorageListener.h"

namespace mitk
{


DataStorageListener::DataStorageListener(mitk::DataStorage::Pointer dataStorage)
: m_DataStorage(NULL)
, m_InDataStorageChanged(false)
{
  assert(dataStorage);
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
}

DataStorageListener::~DataStorageListener()
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
  }
}

void DataStorageListener::NodeAddedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeAdded()
  if(!m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    this->NodeAdded(const_cast<mitk::DataNode*>(node));
    m_InDataStorageChanged = false;
  }
}

void DataStorageListener::NodeChangedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeRemoved()
  if(!m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    this->NodeChanged(const_cast<mitk::DataNode*>(node));
    m_InDataStorageChanged = false;
  }
}

void DataStorageListener::NodeRemovedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeRemoved()
  if(!m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    this->NodeRemoved(const_cast<mitk::DataNode*>(node));
    m_InDataStorageChanged = false;
  }
}

void DataStorageListener::NodeDeletedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is thrown in NodeRemoved()
  if(!m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    this->NodeDeleted(const_cast<mitk::DataNode*>(node));
    m_InDataStorageChanged = false;
  }
}

} // end namespace
