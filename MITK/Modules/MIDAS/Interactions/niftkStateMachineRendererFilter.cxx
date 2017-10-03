/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkStateMachineRendererFilter.h"

#include <mitkBaseRenderer.h>
#include <mitkInteractionEvent.h>

namespace niftk
{

//-----------------------------------------------------------------------------
StateMachineRendererFilter::StateMachineRendererFilter()
{
}


//-----------------------------------------------------------------------------
StateMachineRendererFilter::~StateMachineRendererFilter()
{
}


//-----------------------------------------------------------------------------
bool StateMachineRendererFilter::EventFilter(mitk::InteractionEvent* event) const
{
  mitk::BaseRenderer* renderer = event->GetSender();
  std::vector<mitk::BaseRenderer*>::const_iterator it =
      std::find(m_Renderers.begin(), m_Renderers.end(), renderer);
  return it == m_Renderers.end();
}


//-----------------------------------------------------------------------------
void StateMachineRendererFilter::AddRenderer(mitk::BaseRenderer* renderer)
{
  std::vector<mitk::BaseRenderer*>::iterator it =
      std::find(m_Renderers.begin(), m_Renderers.end(), renderer);

  if (it == m_Renderers.end())
  {
    m_Renderers.push_back(renderer);
  }
}


//-----------------------------------------------------------------------------
void StateMachineRendererFilter::RemoveRenderer(mitk::BaseRenderer* renderer)
{
  std::vector<mitk::BaseRenderer*>::iterator it =
      std::find(m_Renderers.begin(), m_Renderers.end(), renderer);

  if (it != m_Renderers.end())
  {
    m_Renderers.erase(it);
  }
}

}
