/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDataNodeStringPropertyFilter.h"
#include <mitkDataStorage.h>

namespace mitk
{

//-----------------------------------------------------------------------------
DataNodeStringPropertyFilter::DataNodeStringPropertyFilter()
{
  m_Strings.clear();
}


//-----------------------------------------------------------------------------
DataNodeStringPropertyFilter::~DataNodeStringPropertyFilter()
{

}


//-----------------------------------------------------------------------------
void DataNodeStringPropertyFilter::ClearList()
{
  m_Strings.clear();
}


//-----------------------------------------------------------------------------
void DataNodeStringPropertyFilter::AddToList(const std::string& propertyValue)
{
  m_Strings.push_back(propertyValue);
}


//-----------------------------------------------------------------------------
void DataNodeStringPropertyFilter::AddToList(const std::vector< std::string >& listOfStrings)
{
  for (unsigned int i = 0; i < listOfStrings.size(); i++)
  {
    this->AddToList(listOfStrings[i]);
  }
}


//-----------------------------------------------------------------------------
bool DataNodeStringPropertyFilter::Pass(const mitk::DataNode* node)
{
  bool result = true;
  std::string propertyValue;

  if (node != NULL && node->GetStringProperty(this->GetPropertyName().c_str(), propertyValue))
  {
    for (unsigned int i = 0; i < m_Strings.size(); i++)
    {
      if (propertyValue == m_Strings[i])
      {
        result = false;
        break;
      }
    }
  }
  return result;
}

} // end namespace
