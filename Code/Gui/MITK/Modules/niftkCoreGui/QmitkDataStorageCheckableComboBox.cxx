/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkDataStorageCheckableComboBox.h"

#include <itkCommand.h>

//#CTORS/DTOR

QmitkDataStorageCheckableComboBox::QmitkDataStorageCheckableComboBox( QWidget* parent, bool _AutoSelectNewNodes )
: ctkCheckableComboBox(parent)
, m_DataStorage(0)
, m_Predicate(0)
, m_BlockEvents(false)
, m_AutoSelectNewNodes(_AutoSelectNewNodes)
{
  this->Init();
}

QmitkDataStorageCheckableComboBox::QmitkDataStorageCheckableComboBox( mitk::DataStorage* _DataStorage, const mitk::NodePredicateBase* _Predicate,
                                                   QWidget* parent, bool _AutoSelectNewNodes )
: ctkCheckableComboBox(parent)
, m_DataStorage(0)
, m_Predicate(_Predicate)
, m_BlockEvents(false)
, m_AutoSelectNewNodes(_AutoSelectNewNodes)
{
  // make connections, fill combobox
  this->Init();
  this->SetDataStorage(_DataStorage);
}

QmitkDataStorageCheckableComboBox::~QmitkDataStorageCheckableComboBox()
{
  // if there was an old storage, remove listeners
  if(m_DataStorage.IsNotNull())
  {
    this->m_DataStorage->AddNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkDataStorageCheckableComboBox
      , const mitk::DataNode*>( this, &QmitkDataStorageCheckableComboBox::AddNode ) );

    this->m_DataStorage->RemoveNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkDataStorageCheckableComboBox
      , const mitk::DataNode*>( this, &QmitkDataStorageCheckableComboBox::RemoveNode ) );
  }
  //we have lots of observers to nodes and their name properties, this get's ugly if nodes live longer than the box
  while(m_Nodes.size() > 0)
    RemoveNode(0);
}

//#PUBLIC GETTER
mitk::DataStorage::Pointer QmitkDataStorageCheckableComboBox::GetDataStorage() const
{
  return m_DataStorage.GetPointer();
}

const mitk::NodePredicateBase::ConstPointer QmitkDataStorageCheckableComboBox::GetPredicate() const
{
  return m_Predicate.GetPointer();
}

mitk::DataNode::Pointer QmitkDataStorageCheckableComboBox::GetNode( int index ) const
{
  return (this->HasIndex(index))? m_Nodes.at(index): 0;
}

std::vector<mitk::DataNode*> QmitkDataStorageCheckableComboBox::GetSelectedNodes() const
{
  std::vector<mitk::DataNode*> nodes;
  mitk::DataNode* node = NULL;

  QModelIndexList indexes = this->checkedIndexes();

  foreach (QModelIndex item, indexes)
  {
    node = this->GetNode(item.row());
    if (node != NULL)
    {
      nodes.push_back(node);
    }
  }
  return nodes;
}

mitk::DataStorage::SetOfObjects::ConstPointer QmitkDataStorageCheckableComboBox::GetNodes() const
{
  mitk::DataStorage::SetOfObjects::Pointer _SetOfObjects = mitk::DataStorage::SetOfObjects::New();

  for (std::vector<mitk::DataNode*>::const_iterator it = m_Nodes.begin(); it != m_Nodes.end(); ++it)
  {
    _SetOfObjects->push_back(*it);
  }

  return _SetOfObjects.GetPointer();
}

bool QmitkDataStorageCheckableComboBox::GetAutoSelectNewItems()
{
  return m_AutoSelectNewNodes;
}

//#PUBLIC SETTER
void QmitkDataStorageCheckableComboBox::SetDataStorage(mitk::DataStorage* _DataStorage)
{
  // reset only if datastorage really changed
  if(m_DataStorage.GetPointer() != _DataStorage)
  {
    // if there was an old storage, remove listeners
    if(m_DataStorage.IsNotNull())
    {
      this->m_DataStorage->AddNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkDataStorageCheckableComboBox
        , const mitk::DataNode*>( this, &QmitkDataStorageCheckableComboBox::AddNode ) );

      this->m_DataStorage->RemoveNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkDataStorageCheckableComboBox
        , const mitk::DataNode*>( this, &QmitkDataStorageCheckableComboBox::RemoveNode ) );
    }
    // set new storage
    m_DataStorage = _DataStorage;

    // if there is a new storage, add listeners
    if(m_DataStorage.IsNotNull())
    {
      this->m_DataStorage->AddNodeEvent.AddListener( mitk::MessageDelegate1<QmitkDataStorageCheckableComboBox
        , const mitk::DataNode*>( this, &QmitkDataStorageCheckableComboBox::AddNode ) );

      this->m_DataStorage->RemoveNodeEvent.AddListener( mitk::MessageDelegate1<QmitkDataStorageCheckableComboBox
        , const mitk::DataNode*>( this, &QmitkDataStorageCheckableComboBox::RemoveNode ) );
    }

    // reset predicate to reset the combobox
    this->Reset();
  }
}

void QmitkDataStorageCheckableComboBox::SetPredicate(const mitk::NodePredicateBase* _Predicate)
{
  if(m_Predicate != _Predicate)
  {
    m_Predicate = _Predicate;
    this->Reset();
  }
}

void QmitkDataStorageCheckableComboBox::AddNode( const mitk::DataNode* _DataNode )
{
  // this is an event function, make sure that we didnt call ourself
  if(!m_BlockEvents)
  {
    m_BlockEvents = true;
    // pass a -1 to the InsertNode function in order to append the datatreenode to the end
    this->InsertNode(-1, _DataNode);
    m_BlockEvents = false;
  }
}

void QmitkDataStorageCheckableComboBox::RemoveNode( int index )
{
  if(this->HasIndex(index))
  {
    //# remove itk::Event observer
    mitk::DataNode* _DataNode = m_Nodes.at(index);
    // get name property first
    mitk::BaseProperty* nameProperty = _DataNode->GetProperty("name");
    // if prop exists remove modified listener
    if(nameProperty)
    {
      nameProperty->RemoveObserver(m_NodesModifiedObserverTags[index]);
      // remove name property map
      m_PropertyToNode.erase(_DataNode);
    }
    // then remove delete listener on the node itself
    _DataNode->RemoveObserver(m_NodesDeleteObserverTags[index]);
    // remove observer tags from lists
    m_NodesModifiedObserverTags.erase(m_NodesModifiedObserverTags.begin()+index);
    m_NodesDeleteObserverTags.erase(m_NodesDeleteObserverTags.begin()+index);
    // remove node from node vector
    m_Nodes.erase(m_Nodes.begin()+index);
    // remove node name from combobox
    this->removeItem(index);
  }
}

void QmitkDataStorageCheckableComboBox::RemoveNode( const mitk::DataNode* _DataNode )
{
  // this is an event function, make sure that we didnt call ourself
  if(!m_BlockEvents)
  {
    m_BlockEvents = true;
    this->RemoveNode( this->Find(_DataNode) );
    m_BlockEvents = false;
  }
}

void QmitkDataStorageCheckableComboBox::SetNode(int index, const mitk::DataNode* _DataNode)
{
  if(this->HasIndex(index))
  {
    this->InsertNode(index, _DataNode);
  }
}

void QmitkDataStorageCheckableComboBox::SetNode( const mitk::DataNode* _DataNode, const mitk::DataNode* _OtherDataNode)
{
  this->SetNode( this->Find(_DataNode), _OtherDataNode);
}

void QmitkDataStorageCheckableComboBox::SetAutoSelectNewItems( bool _AutoSelectNewItems )
{
  m_AutoSelectNewNodes = _AutoSelectNewItems;
}

void QmitkDataStorageCheckableComboBox::OnDataNodeDeleteOrModified(const itk::Object *caller, const itk::EventObject &event)
{
  if(!m_BlockEvents)
  {
    m_BlockEvents = true;

    // check if we have a modified event (if not it is a delete event)
    const itk::ModifiedEvent* modifiedEvent = dynamic_cast<const itk::ModifiedEvent*>(&event);

    // when node was modified reset text
    if(modifiedEvent)
    {
      const mitk::BaseProperty* _NameProperty = dynamic_cast<const mitk::BaseProperty*>(caller);

      // node name changed, set it
      // but first of all find associated node
      for(std::map<mitk::DataNode*, const mitk::BaseProperty*>::iterator it=m_PropertyToNode.begin()
        ; it!=m_PropertyToNode.end()
        ; ++it)
      {
        // property is found take node
        if(it->second == _NameProperty)
        {
          // looks strange but when calling setnode with the same node, that means the node gets updated
          this->SetNode(it->first, it->first);
          break;
        }
      }
    }
    else
    {
      const mitk::DataNode* _ConstDataNode = dynamic_cast<const mitk::DataNode*>(caller);
      if(_ConstDataNode)
        // node will be deleted, remove it
        this->RemoveNode(_ConstDataNode);
    }

    m_BlockEvents = false;
  }
}

void QmitkDataStorageCheckableComboBox::SetSelectedNode(mitk::DataNode::Pointer item)
{
  int index = this->Find(item);
  if (index == -1)
  {
    MITK_INFO << "QmitkDataStorageCheckableComboBox: item not available";
  }
  else
  {
    this->setCurrentIndex(index);
  }

}

//#PROTECTED GETTER
bool QmitkDataStorageCheckableComboBox::HasIndex(unsigned int index) const
{
  return (m_Nodes.size() > 0 && index < m_Nodes.size());
}

int QmitkDataStorageCheckableComboBox::Find( const mitk::DataNode* _DataNode ) const
{
  int index = -1;

  std::vector<mitk::DataNode*>::const_iterator nodeIt =
    std::find(m_Nodes.begin(), m_Nodes.end(), _DataNode);

  if(nodeIt != m_Nodes.end())
    index = std::distance(m_Nodes.begin(), nodeIt);

  return index;
}


void QmitkDataStorageCheckableComboBox::InsertNode(int index, const mitk::DataNode* _DataNode)
{
  // check new or updated node first
  if(m_Predicate.IsNotNull() && !m_Predicate->CheckNode(_DataNode))
    return;

  bool addNewNode = false;
  bool insertNewNode = false;
  bool changedNode = false;

  // if this->HasIndex(index), then a node shall be updated
  if(this->HasIndex(index))
  {
    // if we really have another node at this position then ...
    if(_DataNode != m_Nodes.at(index))
    {
      // ... remove node, then proceed as usual
      this->RemoveNode(index);
      insertNewNode = true;
    }
    else
      changedNode = true;
  }
  // otherwise a new node shall be added, let index point to the element after the last element
  else
  {
    index = m_Nodes.size();
    addNewNode = true;
  }

  // const cast because we need non const nodes
  mitk::DataNode* _NonConstDataNode = const_cast<mitk::DataNode*>(_DataNode);
  mitk::BaseProperty* nameProperty = _NonConstDataNode->GetProperty("name");

  if(!changedNode)
  {
    // break on duplicated nodes (that doesnt make sense to have duplicates in the combobox)
    if(this->Find(_DataNode) != -1)
      return;

    // add modified observer
    itk::MemberCommand<QmitkDataStorageCheckableComboBox>::Pointer modifiedCommand = itk::MemberCommand<QmitkDataStorageCheckableComboBox>::New();
    modifiedCommand->SetCallbackFunction(this, &QmitkDataStorageCheckableComboBox::OnDataNodeDeleteOrModified);
    // !!!! add modified observer for the name
    /// property of the node because this is the only thing we are interested in !!!!!
    if(nameProperty)
    {
      m_NodesModifiedObserverTags.push_back( nameProperty->AddObserver(itk::ModifiedEvent(), modifiedCommand) );
      m_PropertyToNode[_NonConstDataNode] = nameProperty;
    }
    // if there is no name node save an invalid value for the observer tag (-1)
    else
      m_NodesModifiedObserverTags.push_back( -1 );

    // add delete observer
    itk::MemberCommand<QmitkDataStorageCheckableComboBox>::Pointer deleteCommand = itk::MemberCommand<QmitkDataStorageCheckableComboBox>::New();
    deleteCommand->SetCallbackFunction(this, &QmitkDataStorageCheckableComboBox::OnDataNodeDeleteOrModified);
    m_NodesDeleteObserverTags.push_back( _NonConstDataNode->AddObserver(itk::DeleteEvent(), modifiedCommand) );
  }

  // add node to the vector
  if(addNewNode)
    m_Nodes.push_back( _NonConstDataNode );
  else if(insertNewNode)
    m_Nodes.insert( m_Nodes.begin()+index, _NonConstDataNode );

  // ... and to the combobox
  std::string _NonConstDataNodeName = "unnamed node";
  // _NonConstDataNodeName is "unnamed node" so far, change it if there is a name property in the node
  if(nameProperty)
    _NonConstDataNodeName = nameProperty->GetValueAsString();

  if(addNewNode)
  {
    this->addItem(QString::fromStdString(_NonConstDataNodeName));
    // select new node if m_AutoSelectNewNodes is true or if we have just added the first node
    if(m_AutoSelectNewNodes || m_Nodes.size() == 1)
      this->setCurrentIndex(index);
  }
  else
  {
    // update text in combobox
    this->setItemText( index, QString::fromStdString(_NonConstDataNodeName));
  }
}

void QmitkDataStorageCheckableComboBox::Init()
{
  connect(this, SIGNAL(currentIndexChanged(int)), this, SLOT(OnCurrentIndexChanged(int)));
}

void QmitkDataStorageCheckableComboBox::Reset()
{
  // remove all nodes first
  while( !m_Nodes.empty() )
  {
    // remove last node
    this->RemoveNode( m_Nodes.size() - 1 );
  }

  // clear combobox
  this->clear();

  if(m_DataStorage.IsNotNull())
  {
    mitk::DataStorage::SetOfObjects::ConstPointer setOfObjects;

    // select all if predicate == NULL
    if (m_Predicate.IsNotNull())
      setOfObjects = m_DataStorage->GetSubset(m_Predicate);
    else
      setOfObjects = m_DataStorage->GetAll();

    // add all found nodes
    for (mitk::DataStorage::SetOfObjects::ConstIterator nodeIt = setOfObjects->Begin()
      ; nodeIt != setOfObjects->End(); ++nodeIt)  // for each _DataNode
    {
      // add node to the node vector and to the combobox
      this->AddNode( nodeIt.Value().GetPointer() );
    }
  }
}
