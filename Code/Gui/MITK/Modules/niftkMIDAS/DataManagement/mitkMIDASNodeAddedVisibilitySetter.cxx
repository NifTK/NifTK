/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASNodeAddedVisibilitySetter.h"
#include <mitkDataNode.h>
#include <mitkProperties.h>

namespace mitk
{

//-----------------------------------------------------------------------------
MIDASNodeAddedVisibilitySetter::MIDASNodeAddedVisibilitySetter()
: m_Visibility(false)
, m_Filter(NULL)
{
  m_Filter = mitk::MIDASDataNodeNameStringFilter::New();
  this->AddFilter(m_Filter.GetPointer());
}


//-----------------------------------------------------------------------------
MIDASNodeAddedVisibilitySetter::MIDASNodeAddedVisibilitySetter(mitk::DataStorage::Pointer dataStorage)
: DataStorageListener(dataStorage)
{
}


//-----------------------------------------------------------------------------
MIDASNodeAddedVisibilitySetter::~MIDASNodeAddedVisibilitySetter()
{
}


//-----------------------------------------------------------------------------
void MIDASNodeAddedVisibilitySetter::SetRenderers(std::vector<mitk::BaseRenderer*>& list)
{
  m_Renderers = list;
  this->Modified();
}


//-----------------------------------------------------------------------------
void MIDASNodeAddedVisibilitySetter::ClearRenderers()
{
  m_Renderers.clear();
  this->Modified();
}


//-----------------------------------------------------------------------------
void MIDASNodeAddedVisibilitySetter::NodeAdded(mitk::DataNode* node)
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
