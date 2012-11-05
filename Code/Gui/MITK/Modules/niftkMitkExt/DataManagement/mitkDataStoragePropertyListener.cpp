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

#include "mitkDataStoragePropertyListener.h"
#include <itkCommand.h>

namespace mitk
{

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
  this->UpdateObserverToPropertyMap();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::Deactivate()
{
  this->RemoveAllFromObserverToPropertyMap();
  mitk::DataStorageListener::Deactivate();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::SetPropertyName(const std::string& name)
{
  m_PropertyName = name;
  this->Modified();

  this->UpdateObserverToPropertyMap();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::SetRenderers(std::vector<mitk::BaseRenderer*>& list)
{
  m_Renderers = list;
  this->Modified();

  this->UpdateObserverToPropertyMap();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::ClearRenderers()
{
  m_Renderers.clear();
  this->Modified();

  this->UpdateObserverToPropertyMap();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NodeAdded(mitk::DataNode* node)
{
  this->UpdateObserverToPropertyMap();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NodeRemoved(mitk::DataNode* node)
{
  this->UpdateObserverToPropertyMap();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::NodeDeleted(mitk::DataNode* node)
{
  this->UpdateObserverToPropertyMap();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::RemoveAllFromObserverToPropertyMap()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();

  VectorPropertyToObserver::iterator observerIter = m_WatchedObservers.begin();
  VectorPropertyToNode::iterator nodeIter = m_WatchedNodes.begin();
  for(;
      observerIter != m_WatchedObservers.end();
      ++observerIter,
      ++nodeIter
      )
  {
    // This is bad, as we scan the whole data-storage for each object in our list.
    // I really wanted to do: if (dataStorage->Exists((*nodeIter).second)
    // but if you want to check for whether the node exists, and you pass an invalid pointer,
    // then the Exists method does a std::vector::find, which create a SmartPointer and the Register
    // function crashes, as it appears you cannot create a smart pointer to something that does not exist.
    mitk::DataStorage::SetOfObjects::ConstPointer all = dataStorage->GetAll();
    for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
    {
      if (it->Value() == (*nodeIter).second)
      {
        (*observerIter).first->RemoveObserver((*observerIter).second);
      }
    }
  }
  m_WatchedObservers.clear();
  m_WatchedNodes.clear();

  this->Modified();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::UpdateObserverToPropertyMap()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  if (dataStorage.IsNull())
  {
    return;
  }

  if (m_PropertyName.size() == 0)
  {
    return;
  }

  this->RemoveAllFromObserverToPropertyMap();

  mitk::DataStorage::SetOfObjects::ConstPointer all = dataStorage->GetAll();
  for (mitk::DataStorage::SetOfObjects::ConstIterator it = all->Begin(); it != all->End(); ++it)
  {
    if (it->Value().IsNull() || it->Value()->GetProperty(m_PropertyName.c_str()) == NULL)
    {
      continue;
    }

    bool isHelper = false;
    it->Value()->GetBoolProperty("helper object", isHelper);

    if (isHelper)
    {
      continue;
    }

    /* register listener for changes in property */
    itk::ReceptorMemberCommand<DataStoragePropertyListener>::Pointer command
      = itk::ReceptorMemberCommand<DataStoragePropertyListener>::New();
    command->SetCallbackFunction(this, &DataStoragePropertyListener::OnPropertyChanged);

    mitk::BaseProperty::Pointer property = it->Value()->GetProperty(m_PropertyName.c_str());
    unsigned long observerId = property->AddObserver(itk::ModifiedEvent(), command);
    PropertyToObserver tag(property, observerId);
    PropertyToNode node(property, it->Value());

    m_WatchedObservers.push_back(tag);
    m_WatchedNodes.push_back(node);

    for (unsigned int i = 0; i < m_Renderers.size(); i++)
    {
      itk::ReceptorMemberCommand<DataStoragePropertyListener>::Pointer rendererSpecificCommand
        = itk::ReceptorMemberCommand<DataStoragePropertyListener>::New();
      rendererSpecificCommand->SetCallbackFunction(this, &DataStoragePropertyListener::OnPropertyChanged);

      mitk::BaseProperty::Pointer rendererSpecificProperty = it->Value()->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);
      unsigned long rendererSpecificObserverId = rendererSpecificProperty->AddObserver(itk::ModifiedEvent(), command);

      PropertyToObserver rendererSpecificTag(rendererSpecificProperty, rendererSpecificObserverId);
      m_WatchedObservers.push_back(rendererSpecificTag);

      PropertyToNode rendererSpecificNode(rendererSpecificProperty, it->Value());
      m_WatchedNodes.push_back(rendererSpecificNode);
    }

    this->OnPropertyChanged();
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::OnPropertyChanged(const itk::EventObject&)
{
  this->OnPropertyChanged();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::OnPropertyChanged()
{
  if (!this->GetBlock())
  {
    PropertyChanged.Send();
  }
}

} // end namespace
