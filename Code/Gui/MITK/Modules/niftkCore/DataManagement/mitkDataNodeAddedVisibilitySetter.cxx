/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDataNodeAddedVisibilitySetter.h"
#include <mitkDataNode.h>
#include <mitkProperties.h>

namespace mitk
{

//-----------------------------------------------------------------------------
DataNodeAddedVisibilitySetter::DataNodeAddedVisibilitySetter()
: m_Visibility(false)
{
}


//-----------------------------------------------------------------------------
DataNodeAddedVisibilitySetter::DataNodeAddedVisibilitySetter(mitk::DataStorage::Pointer dataStorage)
: DataStorageListener(dataStorage)
{
}


//-----------------------------------------------------------------------------
DataNodeAddedVisibilitySetter::~DataNodeAddedVisibilitySetter()
{
}


//-----------------------------------------------------------------------------
void DataNodeAddedVisibilitySetter::SetRenderers(std::vector<mitk::BaseRenderer*>& list)
{
  m_Renderers = list;
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataNodeAddedVisibilitySetter::ClearRenderers()
{
  m_Renderers.clear();
  this->Modified();
}


//-----------------------------------------------------------------------------
void DataNodeAddedVisibilitySetter::NodeAdded(mitk::DataNode* node)
{
  if (m_Renderers.size() > 0)
  {
    for (unsigned int i = 0; i < m_Renderers.size(); i++)
    {
      node->SetBoolProperty("visible", m_Visibility, m_Renderers[i]);
    }
  }
  else
  {
    node->SetBoolProperty("visible", m_Visibility);
  }
}

} // end namespace
