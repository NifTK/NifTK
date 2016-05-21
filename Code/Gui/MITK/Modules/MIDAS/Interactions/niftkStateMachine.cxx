/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkStateMachine.h"

#include <itkCommand.h>

#include <mitkGlobalInteraction.h>
#include <mitkToolManager.h>

// MicroServices
#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleRegistry.h>

#include "niftkEventFilter.h"

namespace niftk
{

//-----------------------------------------------------------------------------
MIDASStateMachine::MIDASStateMachine()
{
}


//-----------------------------------------------------------------------------
MIDASStateMachine::~MIDASStateMachine()
{
}


//-----------------------------------------------------------------------------
float MIDASStateMachine::CanHandleEvent(const mitk::StateEvent* stateEvent) const
{
  if (this->IsFiltered(stateEvent))
  {
    return 0.0;
  }

  return this->CanHandle(stateEvent);
}


//-----------------------------------------------------------------------------
bool MIDASStateMachine::CanHandleEvent(mitk::InteractionEvent* event)
{
  if (this->IsFiltered(event))
  {
    return false;
  }

  return this->CanHandle(event);
}


//-----------------------------------------------------------------------------
void MIDASStateMachine::InstallEventFilter(MIDASEventFilter* eventFilter)
{
  std::vector<MIDASEventFilter*>::iterator it =
      std::find(m_EventFilters.begin(), m_EventFilters.end(), eventFilter);

  if (it == m_EventFilters.end())
  {
    m_EventFilters.push_back(eventFilter);
  }
}


//-----------------------------------------------------------------------------
void MIDASStateMachine::RemoveEventFilter(MIDASEventFilter* eventFilter)
{
  std::vector<MIDASEventFilter*>::iterator it =
      std::find(m_EventFilters.begin(), m_EventFilters.end(), eventFilter);

  if (it != m_EventFilters.end())
  {
    m_EventFilters.erase(it);
  }
}


//-----------------------------------------------------------------------------
std::vector<MIDASEventFilter*> MIDASStateMachine::GetEventFilters() const
{
  return m_EventFilters;
}


//-----------------------------------------------------------------------------
bool MIDASStateMachine::IsFiltered(const mitk::StateEvent* stateEvent) const
{
  /// Sanity check.
  if (!stateEvent || !stateEvent->GetEvent()->GetSender())
  {
    return true;
  }

  std::vector<MIDASEventFilter*>::const_iterator it = m_EventFilters.begin();
  std::vector<MIDASEventFilter*>::const_iterator itEnd = m_EventFilters.end();

  for ( ; it != itEnd; ++it)
  {
    if ((*it)->EventFilter(stateEvent))
    {
      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------------
bool MIDASStateMachine::IsFiltered(mitk::InteractionEvent* event)
{
  /// Sanity check.
  if (!event || !event->GetSender())
  {
    return true;
  }

  std::vector<MIDASEventFilter*>::const_iterator it = m_EventFilters.begin();
  std::vector<MIDASEventFilter*>::const_iterator itEnd = m_EventFilters.end();

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
