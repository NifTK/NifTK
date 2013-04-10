/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDataNodeBoolPropertyFilter.h"
#include <mitkBaseRenderer.h>
#include <mitkDataStorage.h>

namespace mitk
{

//-----------------------------------------------------------------------------
DataNodeBoolPropertyFilter::DataNodeBoolPropertyFilter()
: m_DataStorage(NULL)
{
  m_Renderers.clear();
}


//-----------------------------------------------------------------------------
DataNodeBoolPropertyFilter::~DataNodeBoolPropertyFilter()
{

}


//-----------------------------------------------------------------------------
void DataNodeBoolPropertyFilter::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  m_DataStorage = dataStorage;
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataNodeBoolPropertyFilter::SetRenderers(std::vector<mitk::BaseRenderer*>& list)
{
  m_Renderers = list;
  this->Modified();
}


//-----------------------------------------------------------------------------
bool DataNodeBoolPropertyFilter::Pass(const mitk::DataNode* node)
{
  bool result = false; // block by default.

  if (m_DataStorage.IsNotNull() && node != NULL && m_PropertyName.length() > 0)
  {
    bool globallyTrue = false;
    bool rendererSpecificPropertyIsTrue = false;

    node->GetBoolProperty(m_PropertyName.c_str(), globallyTrue);

    for (unsigned int i = 0; i < m_Renderers.size(); i++)
    {
      bool foundRendererSpecificProperty = false;
      bool tmp = false;

      foundRendererSpecificProperty = node->GetBoolProperty(m_PropertyName.c_str(), tmp, m_Renderers[i]);
      if (foundRendererSpecificProperty)
      {
        if (tmp)
        {
          rendererSpecificPropertyIsTrue = true;
        }
      }
    }

    result = (globallyTrue || (!globallyTrue && rendererSpecificPropertyIsTrue));
  }

  return result;
}

} // end namespace
