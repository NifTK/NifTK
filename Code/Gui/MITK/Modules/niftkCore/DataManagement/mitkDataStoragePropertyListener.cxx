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
  mitkNewMacro3Param(VisibilityChangedCommand, DataStoragePropertyListener*, mitk::DataNode*, const mitk::BaseRenderer*);

  VisibilityChangedCommand(DataStoragePropertyListener* observer, mitk::DataNode* node, const mitk::BaseRenderer* renderer)
  : m_Observer(observer)
  , m_Node(node)
  , m_Renderer(renderer)
  {
    assert(observer);
    assert(node);
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
  const mitk::BaseRenderer* m_Renderer;
};


//-----------------------------------------------------------------------------
DataStoragePropertyListener::DataStoragePropertyListener(const std::string& propertyName)
: m_PropertyName(propertyName)
{
}


//-----------------------------------------------------------------------------
DataStoragePropertyListener::~DataStoragePropertyListener()
{
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::Activate(const mitk::DataStorage::Pointer dataStorage)
{
  Superclass::Activate(dataStorage);
  this->AddAllObservers();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::Deactivate()
{
  this->RemoveAllObservers();
  mitk::DataStorageListener::Deactivate();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::SetRenderers(const std::vector<const mitk::BaseRenderer*>& renderers)
{
  this->RemoveAllObservers();

  m_Renderers = renderers;

  this->AddAllObservers();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NodeAdded(mitk::DataNode* node)
{
  this->AddObservers(node);
  this->Notify(node);
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NodeRemoved(mitk::DataNode* node)
{
  this->Notify(node);
  this->RemoveObservers(node);
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NodeDeleted(mitk::DataNode* node)
{
  this->Notify(node);
  this->RemoveObservers(node);
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

  mitk::BaseProperty* globalProperty = node->GetProperty(m_PropertyName.c_str());
  if (globalProperty)
  {
    VisibilityChangedCommand::Pointer command = VisibilityChangedCommand::New(this, node, 0);
    propertyObserverTags[0] = globalProperty->AddObserver(itk::AnyEvent(), command);
  }
  else
  {
    propertyObserverTags[0] = 0;
  }

  for (std::size_t i = 0; i < m_Renderers.size(); ++i)
  {
    /// Note:
    /// GetProperty() returns the global property if there is no renderer specific property.
    /// Therefore, we need to check if the property is really renderer specific.
    mitk::BaseProperty* rendererSpecificProperty = node->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);
    if (rendererSpecificProperty && rendererSpecificProperty != globalProperty)
    {
      VisibilityChangedCommand::Pointer command = VisibilityChangedCommand::New(this, node, m_Renderers[i]);
      propertyObserverTags[i + 1] = rendererSpecificProperty->AddObserver(itk::AnyEvent(), command);
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
  NodePropertyObserverTags::iterator propertyObserverTagsPerNodeIt = m_PropertyObserverTagsPerNode.find(node);
  if (propertyObserverTagsPerNodeIt != m_PropertyObserverTagsPerNode.end())
  {
    std::vector<unsigned long>& propertyObserverTags = propertyObserverTagsPerNodeIt->second;
    mitk::BaseProperty* globalProperty = node->GetProperty(m_PropertyName.c_str(), 0);
    if (globalProperty)
    {
      globalProperty->RemoveObserver(propertyObserverTags[0]);
      propertyObserverTags[0] = 0;
    }

    for (std::size_t i = 0; i < m_Renderers.size(); ++i)
    {
      /// Note:
      /// GetProperty() returns the global property if there is no renderer specific property.
      /// Therefore, we need to check if the property is really renderer specific.
      mitk::BaseProperty* rendererSpecificProperty = node->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);
      if (rendererSpecificProperty && rendererSpecificProperty != globalProperty)
      {
        rendererSpecificProperty->RemoveObserver(propertyObserverTags[i + 1]);
        propertyObserverTags[i + 1] = 0;
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
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::RemoveAllObservers()
{
  NodePropertyObserverTags::iterator propertyObserverTagsIt = m_PropertyObserverTagsPerNode.begin();
  NodePropertyObserverTags::iterator nodeToObserverTagsEnd = m_PropertyObserverTagsPerNode.end();

  for ( ; propertyObserverTagsIt != nodeToObserverTagsEnd; ++propertyObserverTagsIt)
  {
    mitk::DataNode* node = propertyObserverTagsIt->first;
    std::vector<unsigned long>& propertyObserverTags = propertyObserverTagsIt->second;
    if (propertyObserverTags.empty())
    {
      continue;
    }

    mitk::BaseProperty* globalProperty = node->GetProperty(m_PropertyName.c_str(), 0);
    if (globalProperty)
    {
      globalProperty->RemoveObserver(propertyObserverTags[0]);
    }

    for (std::size_t i = 0; i < m_Renderers.size(); ++i)
    {
      /// Note:
      /// GetProperty() returns the global property if there is no renderer specific property.
      /// Therefore, we need to check if the property is really renderer specific.
      mitk::BaseProperty* rendererSpecificProperty = node->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);
      if (rendererSpecificProperty && rendererSpecificProperty != globalProperty)
      {
        rendererSpecificProperty->RemoveObserver(propertyObserverTags[i + 1]);
      }
    }
  }

  m_PropertyObserverTagsPerNode.clear();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::OnPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  if (!this->IsBlocked())
  {
    PropertyChanged.Send(node, renderer);
  }
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::Notify(mitk::DataNode* node)
{
  NodePropertyObserverTags::iterator propertyObserverTagsPerNodeIt = m_PropertyObserverTagsPerNode.find(node);
  if (propertyObserverTagsPerNodeIt != m_PropertyObserverTagsPerNode.end())
  {
    std::vector<unsigned long>& propertyObserverTags = propertyObserverTagsPerNodeIt->second;

    mitk::BaseProperty* globalProperty = node->GetProperty(m_PropertyName.c_str(), 0);
    if (globalProperty && propertyObserverTags[0])
    {
      VisibilityChangedCommand* observer = dynamic_cast<VisibilityChangedCommand*>(globalProperty->GetCommand(propertyObserverTags[0]));
      /// Note:
      /// We need to do a null check here because the observer tag is not cleared when a property is removed.
      if (observer)
      {
        observer->Notify();
      }
    }

    for (std::size_t i = 0; i < m_Renderers.size(); ++i)
    {
      /// Note:
      /// GetProperty() returns the global property if there is no renderer specific property.
      /// Therefore, we need to check if the property is really renderer specific.
      mitk::BaseProperty* rendererSpecificProperty = node->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);
      if (rendererSpecificProperty && rendererSpecificProperty != globalProperty && propertyObserverTags[i + 1])
      {
        VisibilityChangedCommand* observer = dynamic_cast<VisibilityChangedCommand*>(rendererSpecificProperty->GetCommand(propertyObserverTags[i + 1]));
        /// Note:
        /// We need to do a null check here because the observer tag is not cleared when a property is removed.
        if (observer)
        {
          observer->Notify();
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NotifyAll()
{
  if (!this->IsBlocked())
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
