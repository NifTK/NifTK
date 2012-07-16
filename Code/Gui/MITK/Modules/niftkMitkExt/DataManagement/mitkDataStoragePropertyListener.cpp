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
: m_DataStorage(NULL)
, m_PropertyName("")
{
  m_ObserverToPropertyMap.clear();
}


//-----------------------------------------------------------------------------
DataStoragePropertyListener::DataStoragePropertyListener(const mitk::DataStorage::Pointer dataStorage)
: m_DataStorage(dataStorage)
, m_PropertyName("")
{
  m_ObserverToPropertyMap.clear();
  this->Activate(dataStorage);
}


//-----------------------------------------------------------------------------
DataStoragePropertyListener::~DataStoragePropertyListener()
{
  this->Deactivate();
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::SetDataStorage(const mitk::DataStorage::Pointer dataStorage)
{
  this->Activate(dataStorage);
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::Activate(const mitk::DataStorage::Pointer dataStorage)
{
  if (this->m_DataStorage.IsNotNull())
  {
    this->Deactivate();
  }

  if (dataStorage.IsNotNull())
  {
    m_DataStorage = dataStorage;

    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void DataStoragePropertyListener::Deactivate()
{
  if (m_DataStorage.IsNotNull())
  {
    this->RemoveAllFromObserverToPropertyMap();
    this->Modified();
  }
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

    /* register listener for changes in visible property */
    itk::ReceptorMemberCommand<DataStoragePropertyListener>::Pointer command
      = itk::ReceptorMemberCommand<DataStoragePropertyListener>::New();
    command->SetCallbackFunction(this, &DataStoragePropertyListener::OnPropertyChanged);
    m_ObserverToPropertyMap[it->Value()->GetProperty(m_PropertyName.c_str())->AddObserver( itk::ModifiedEvent(), command )]
                            = it->Value()->GetProperty(m_PropertyName.c_str());
  }

}

} // end namespace
