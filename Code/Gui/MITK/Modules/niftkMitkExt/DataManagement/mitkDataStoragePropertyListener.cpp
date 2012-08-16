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
  m_WatchedNodes.clear();
}


//-----------------------------------------------------------------------------
DataStoragePropertyListener::DataStoragePropertyListener(const mitk::DataStorage::Pointer dataStorage)
: mitk::DataStorageListener(dataStorage)
, m_PropertyName("")
{
  m_WatchedNodes.clear();
}


//-----------------------------------------------------------------------------
DataStoragePropertyListener::~DataStoragePropertyListener()
{
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
  mitk::DataStorageListener::Deactivate();
  this->RemoveAllFromObserverToPropertyMap();
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
  for( VectorPropertyToObserver::iterator iter = m_WatchedNodes.begin();
      iter != m_WatchedNodes.end(); ++iter )
  {
    (*iter).first->RemoveObserver((*iter).second);
  }
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
    m_WatchedNodes.push_back(tag);

    for (unsigned int i = 0; i < m_Renderers.size(); i++)
    {
      itk::ReceptorMemberCommand<DataStoragePropertyListener>::Pointer rendererSpecificCommand
        = itk::ReceptorMemberCommand<DataStoragePropertyListener>::New();
      rendererSpecificCommand->SetCallbackFunction(this, &DataStoragePropertyListener::OnPropertyChanged);

      mitk::BaseProperty::Pointer rendererSpecificProperty = it->Value()->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);
      unsigned long rendererSpecificObserverId = rendererSpecificProperty->AddObserver(itk::ModifiedEvent(), command);
      PropertyToObserver rendererSpecificTag(rendererSpecificProperty, rendererSpecificObserverId);
      m_WatchedNodes.push_back(rendererSpecificTag);
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
