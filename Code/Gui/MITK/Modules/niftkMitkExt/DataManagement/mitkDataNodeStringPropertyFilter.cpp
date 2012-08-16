/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-17 12:27:28 +0100 (Tue, 17 Jul 2012) $
 Revision          : $Revision: 9362 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
