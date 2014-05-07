/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDataStoragePropertyListener.h"
#include <itkCommand.h>

#include <mitkBaseRenderer.h>

namespace mitk
{

class VisibilityChangedCommand : public itk::Command
{
public:
  mitkClassMacro(VisibilityChangedCommand, itk::Command);
  mitkNewMacro3Param(VisibilityChangedCommand, DataStoragePropertyListener*, mitk::DataNode*, mitk::BaseRenderer*);

  VisibilityChangedCommand(DataStoragePropertyListener* observer, mitk::DataNode* node, mitk::BaseRenderer* renderer)
  : m_Observer(observer)
  , m_Node(node)
  , m_Renderer(renderer)
  {
  }

  virtual ~VisibilityChangedCommand()
  {
  }

  virtual void Execute(itk::Object* /*caller*/, const itk::EventObject& /*event*/)
  {
    this->Notify();
  }

  virtual void Execute(const itk::Object* /*caller*/, const itk::EventObject& /*event*/)
  {
    this->Notify();
  }

  void Notify()
  {
    m_Observer->OnPropertyChanged(m_Node, m_Renderer);
  }

private:
  DataStoragePropertyListener* m_Observer;
  mitk::DataNode* m_Node;
  mitk::BaseRenderer* m_Renderer;
};

//-----------------------------------------------------------------------------
DataStoragePropertyListener::DataStoragePropertyListener()
: m_PropertyName("")
{
}


//-----------------------------------------------------------------------------
DataStoragePropertyListener::DataStoragePropertyListener(const mitk::DataStorage::Pointer dataStorage)
: mitk::DataStorageListener(dataStorage)
, m_PropertyName("")
{
}


//-----------------------------------------------------------------------------
DataStoragePropertyListener::~DataStoragePropertyListener()
{
  this->Deactivate();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::Activate(const mitk::DataStorage::Pointer dataStorage)
{
  mitk::DataStorageListener::Activate(dataStorage);
  this->AddAllObservers();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::Deactivate()
{
  this->RemoveAllObservers();
  mitk::DataStorageListener::Deactivate();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::SetPropertyName(const std::string& name)
{
  this->RemoveAllObservers();

  m_PropertyName = name;

  this->AddAllObservers();

  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::SetRenderers(const std::vector<mitk::BaseRenderer*>& renderers)
{
  this->RemoveAllObservers();

  m_Renderers = renderers;

  this->AddAllObservers();

  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NodeAdded(mitk::DataNode* node)
{
  this->AddObservers(node);
  this->Notify(node);
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NodeRemoved(mitk::DataNode* node)
{
  this->RemoveObservers(node);
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NodeDeleted(mitk::DataNode* node)
{
  this->RemoveObservers(node);
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::AddObservers(mitk::DataNode* node)
{
  if (!node)
  {
    return;
  }

  /// Note:
  /// We register the property observers to itk::AnyEvent that includes
  /// itk::ModifiedEvent and itk::DeleteEvent as well.

  std::vector<unsigned long> propertyObserverTags(m_Renderers.size() + 1);

  mitk::BaseProperty* property = node->GetProperty(m_PropertyName.c_str());
  if (property)
  {
    VisibilityChangedCommand::Pointer command = VisibilityChangedCommand::New(this, node, 0);
    propertyObserverTags[0] = property->AddObserver(itk::AnyEvent(), command);
  }
  else
  {
    propertyObserverTags[0] = 0;
  }

  for (std::size_t i = 0; i < m_Renderers.size(); ++i)
  {
    property = node->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);
    if (property)
    {
      VisibilityChangedCommand::Pointer command = VisibilityChangedCommand::New(this, node, 0);
      propertyObserverTags[i + 1] = property->AddObserver(itk::AnyEvent(), command);
    }
    else
    {
      propertyObserverTags[i + 1] = 0;
    }
  }

  m_PropertyObserverTagsPerNode[node] = propertyObserverTags;
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::RemoveObservers(mitk::DataNode* node)
{
  std::vector<unsigned long>& propertyObserverTags = m_PropertyObserverTagsPerNode[node];
  if (!propertyObserverTags.empty())
  {
    mitk::BaseProperty* property = node->GetProperty(m_PropertyName.c_str(), 0);
    if (property)
    {
      property->RemoveObserver(propertyObserverTags[0]);
    }

    for (std::size_t i = 0; i < m_Renderers.size(); ++i)
    {
      property = node->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);
      if (property)
      {
        property->RemoveObserver(propertyObserverTags[i + 1]);
      }
    }

    m_PropertyObserverTagsPerNode.erase(node);
  }
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::AddAllObservers()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  if (dataStorage.IsNull())
  {
    return;
  }

  mitk::DataStorage::SetOfObjects::ConstPointer all = dataStorage->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    this->AddObservers(it->Value().GetPointer());
  }

  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::RemoveAllObservers()
{
  NodePropertyObserverTags::iterator propertyObserverTagsIt = m_PropertyObserverTagsPerNode.begin();
  NodePropertyObserverTags::iterator nodeToObserverTagsEnd = m_PropertyObserverTagsPerNode.end();

  for ( ; propertyObserverTagsIt != nodeToObserverTagsEnd; ++propertyObserverTagsIt)
  {
    mitk::DataNode* node = propertyObserverTagsIt->first;
    std::vector<unsigned long>& observerTags = propertyObserverTagsIt->second;
    if (observerTags.empty())
    {
      continue;
    }

    mitk::BaseProperty* property = node->GetProperty(m_PropertyName.c_str(), 0);
    if (property)
    {
      property->RemoveObserver(observerTags[0]);
    }
    for (std::size_t i = 0; i < m_Renderers.size(); ++i)
    {
      property = node->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);
      if (property)
      {
        property->RemoveObserver(observerTags[i + 1]);
      }
    }
  }

  m_PropertyObserverTagsPerNode.clear();

  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::OnPropertyChanged(mitk::DataNode* node, mitk::BaseRenderer* renderer)
{
  if (!this->GetBlocked())
  {
    PropertyChanged.Send(node, renderer);
  }
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::Notify(mitk::DataNode* node)
{
  std::vector<unsigned long>& observerTags = m_PropertyObserverTagsPerNode[node];
  if (observerTags.empty())
  {
    return;
  }

  mitk::BaseProperty* property = node->GetProperty(m_PropertyName.c_str(), 0);
  if (property && observerTags[0])
  {
    VisibilityChangedCommand* observer = dynamic_cast<VisibilityChangedCommand*>(property->GetCommand(observerTags[0]));
    observer->Notify();
  }
  for (std::size_t i = 0; i < m_Renderers.size(); ++i)
  {
    property = node->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);
    if (property)
    {
      VisibilityChangedCommand* observer = dynamic_cast<VisibilityChangedCommand*>(property->GetCommand(observerTags[i + 1]));
      observer->Notify();
    }
  }
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NotifyAll()
{
  if (!this->GetBlocked())
  {
    NodePropertyObserverTags::iterator nodeToObserverTagsIt = m_PropertyObserverTagsPerNode.begin();
    NodePropertyObserverTags::iterator nodeToObserverTagsEnd = m_PropertyObserverTagsPerNode.end();

    for ( ; nodeToObserverTagsIt != nodeToObserverTagsEnd; ++nodeToObserverTagsIt)
    {
      this->Notify(nodeToObserverTagsIt->first);
    }
  }
}

} // end namespace
