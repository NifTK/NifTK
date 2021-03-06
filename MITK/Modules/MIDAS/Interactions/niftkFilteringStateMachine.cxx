/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkFilteringStateMachine.h"

#include <itkCommand.h>

#include <mitkGlobalInteraction.h>
#include <mitkToolManager.h>

// MicroServices
#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleRegistry.h>

#include "niftkStateMachineEventFilter.h"

namespace niftk
{

//-----------------------------------------------------------------------------
FilteringStateMachine::FilteringStateMachine()
{
}


//-----------------------------------------------------------------------------
FilteringStateMachine::~FilteringStateMachine()
{
}


//-----------------------------------------------------------------------------
bool FilteringStateMachine::CanHandleEvent(mitk::InteractionEvent* event)
{
  if (this->IsFiltered(event))
  {
    return false;
  }

  return this->CanHandle(event);
}


//-----------------------------------------------------------------------------
void FilteringStateMachine::InstallEventFilter(StateMachineEventFilter* eventFilter)
{
  std::vector<StateMachineEventFilter*>::iterator it =
      std::find(m_EventFilters.begin(), m_EventFilters.end(), eventFilter);

  if (it == m_EventFilters.end())
  {
    m_EventFilters.push_back(eventFilter);
  }
}


//-----------------------------------------------------------------------------
void FilteringStateMachine::RemoveEventFilter(StateMachineEventFilter* eventFilter)
{
  std::vector<StateMachineEventFilter*>::iterator it =
      std::find(m_EventFilters.begin(), m_EventFilters.end(), eventFilter);

  if (it != m_EventFilters.end())
  {
    m_EventFilters.erase(it);
  }
}


//-----------------------------------------------------------------------------
std::vector<StateMachineEventFilter*> FilteringStateMachine::GetEventFilters() const
{
  return m_EventFilters;
}


//-----------------------------------------------------------------------------
bool FilteringStateMachine::IsFiltered(mitk::InteractionEvent* event)
{
  /// Sanity check.
  if (!event || !event->GetSender())
  {
    return true;
  }

  std::vector<StateMachineEventFilter*>::const_iterator it = m_EventFilters.begin();
  std::vector<StateMachineEventFilter*>::const_iterator itEnd = m_EventFilters.end();

  for ( ; it != itEnd; ++it)
  {
    if ((*it)->EventFilter(event))
    {
      return true;
    }
  }

  return false;
}

}
