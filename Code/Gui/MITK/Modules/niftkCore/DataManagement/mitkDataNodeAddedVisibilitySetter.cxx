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
bool DataNodeAddedVisibilitySetter::GetVisibility() const
{
  return m_Visibility;
}


//-----------------------------------------------------------------------------
void DataNodeAddedVisibilitySetter::SetVisibility(bool visibility)
{
  m_Visibility = visibility;
}


//-----------------------------------------------------------------------------
void DataNodeAddedVisibilitySetter::SetRenderers(const std::vector<const mitk::BaseRenderer*>& renderers)
{
  m_Renderers = renderers;
}


//-----------------------------------------------------------------------------
void DataNodeAddedVisibilitySetter::ClearRenderers()
{
  m_Renderers.clear();
}


//-----------------------------------------------------------------------------
void DataNodeAddedVisibilitySetter::NodeAdded(mitk::DataNode* node)
{
  if (m_Renderers.size() > 0)
  {
    /// TODO
//    node->SetBoolProperty("visible", m_Visibility);
    for (unsigned int i = 0; i < m_Renderers.size(); i++)
    {
      /// TODO
      /// The const_cast is needed because of the MITK bug 17778. It should be removed after the bug is fixed.
      node->SetBoolProperty("visible", m_Visibility, const_cast<mitk::BaseRenderer*>(m_Renderers[i]));
    }
  }
  else
  {
    node->SetBoolProperty("visible", m_Visibility);
  }
}

} // end namespace
