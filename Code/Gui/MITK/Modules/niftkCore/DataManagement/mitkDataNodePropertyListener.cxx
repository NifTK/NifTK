/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDataNodePropertyListener.h"
#include <itkCommand.h>

#include <mitkBaseRenderer.h>

namespace mitk
{

class PropertyChangedCommand : public itk::Command
{
public:
  mitkClassMacro(PropertyChangedCommand, itk::Command);
  mitkNewMacro3Param(PropertyChangedCommand, DataNodePropertyListener*, mitk::DataNode*, const mitk::BaseRenderer*);

  PropertyChangedCommand(DataNodePropertyListener* observer, mitk::DataNode* node, const mitk::BaseRenderer* renderer)
  : m_Observer(observer)
  , m_Node(node)
  , m_Renderer(renderer)
  {
    assert(observer);
    assert(node);
  }

  virtual ~PropertyChangedCommand()
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
   if (!m_Observer->IsBlocked())
   {
      m_Observer->OnPropertyChanged(m_Node, m_Renderer);
   }
  }

private:
  DataNodePropertyListener* m_Observer;
  mitk::DataNode* m_Node;
  const mitk::BaseRenderer* m_Renderer;
};


//-----------------------------------------------------------------------------
DataNodePropertyListener::DataNodePropertyListener(const mitk::DataStorage::Pointer dataStorage, const std::string& propertyName)
: mitk::DataStorageListener(dataStorage),
  m_PropertyName(propertyName)
{
  assert(dataStorage.IsNotNull());

  this->AddAllObservers();
}


//-----------------------------------------------------------------------------
DataNodePropertyListener::~DataNodePropertyListener()
{
  this->RemoveAllObservers();
}


//-----------------------------------------------------------------------------
void DataNodePropertyListener::SetRenderers(const std::vector<const mitk::BaseRenderer*>& renderers)
{
  if (renderers != m_Renderers)
  {
    this->RemoveAllObservers();

    m_Renderers = renderers;

    this->AddAllObservers();
  }
}


//-----------------------------------------------------------------------------
void DataNodePropertyListener::OnNodeAdded(mitk::DataNode* node)
{
  Superclass::OnNodeAdded(node);
  this->AddObservers(node);
}


//-----------------------------------------------------------------------------
void DataNodePropertyListener::OnNodeRemoved(mitk::DataNode* node)
{
  Superclass::OnNodeAdded(node);
  this->RemoveObservers(node);
}


//-----------------------------------------------------------------------------
void DataNodePropertyListener::OnNodeDeleted(mitk::DataNode* node)
{
  Superclass::OnNodeAdded(node);
  this->RemoveObservers(node);
}


//-----------------------------------------------------------------------------
void DataNodePropertyListener::AddObservers(mitk::DataNode* node)
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
    PropertyChangedCommand::Pointer command = PropertyChangedCommand::New(this, node, 0);
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
      PropertyChangedCommand::Pointer command = PropertyChangedCommand::New(this, node, m_Renderers[i]);
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
void DataNodePropertyListener::RemoveObservers(mitk::DataNode* node)
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
void DataNodePropertyListener::AddAllObservers()
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
void DataNodePropertyListener::RemoveAllObservers()
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
void DataNodePropertyListener::OnPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
}

}
