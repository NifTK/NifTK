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
  m_ObserverToPropertyMap.clear();
}


//-----------------------------------------------------------------------------
DataStoragePropertyListener::DataStoragePropertyListener(const mitk::DataStorage::Pointer dataStorage)
: mitk::DataStorageListener(dataStorage)
, m_PropertyName("")
{
  m_ObserverToPropertyMap.clear();
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
  for( std::map<unsigned long, mitk::BaseProperty::Pointer>::iterator iter = m_ObserverToPropertyMap.begin();
      iter != m_ObserverToPropertyMap.end(); ++iter )
  {
    (*iter).second->RemoveObserver((*iter).first);
  }
  m_ObserverToPropertyMap.clear();
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

    m_ObserverToPropertyMap[it->Value()->GetProperty(m_PropertyName.c_str())->AddObserver( itk::ModifiedEvent(), command )]
                            = it->Value()->GetProperty(m_PropertyName.c_str());

    for (unsigned int i = 0; i < m_Renderers.size(); i++)
    {
      itk::ReceptorMemberCommand<DataStoragePropertyListener>::Pointer rendererSpecificCommand
        = itk::ReceptorMemberCommand<DataStoragePropertyListener>::New();
      rendererSpecificCommand->SetCallbackFunction(this, &DataStoragePropertyListener::OnPropertyChanged);

      m_ObserverToPropertyMap[it->Value()->GetProperty(m_PropertyName.c_str(), m_Renderers[i])->AddObserver( itk::ModifiedEvent(), command )]
                              = it->Value()->GetProperty(m_PropertyName.c_str(), m_Renderers[i]);

    }
    this->Modified();
  }

}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::OnPropertyChanged(const itk::EventObject&)
{
  if (!this->GetBlock())
  {
    PropertyChanged.Send();
  }
}

} // end namespace
