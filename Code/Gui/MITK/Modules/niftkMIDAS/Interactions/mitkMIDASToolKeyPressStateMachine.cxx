/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASToolKeyPressStateMachine.h"
#include <mitkWheelEvent.h>
#include <mitkStateEvent.h>

namespace mitk
{

//-----------------------------------------------------------------------------
MIDASToolKeyPressStateMachine::MIDASToolKeyPressStateMachine(const char * stateMachinePattern, MIDASToolKeyPressResponder* responder)
: StateMachine(stateMachinePattern)
{
  assert(responder);
  m_Responder = responder;

  CONNECT_ACTION( 350006, SelectSeedTool );
  CONNECT_ACTION( 350007, SelectDrawTool );
  CONNECT_ACTION( 350008, UnselectTools );
  CONNECT_ACTION( 350009, UnselectTools );
  CONNECT_ACTION( 350010, SelectPolyTool );
  CONNECT_ACTION( 350011, SelectViewMode );
  CONNECT_ACTION( 350012, CleanSlice );
}


//-----------------------------------------------------------------------------
float MIDASToolKeyPressStateMachine::CanHandleEvent(const StateEvent *event) const
{
  // See StateMachine.xml for event Ids.
  if (event != NULL
      && event->GetEvent() != NULL
      && (   event->GetId() == 18   // S
          || event->GetId() == 4004 // D
          || event->GetId() == 4018 // Y
          || event->GetId() == 4015 // V
          || event->GetId() == 4003 // C
          || event->GetId() == 13   // N
          || event->GetId() == 25   // Space
          )
      )
  {
    return 1;
  }
  else
  {
    return mitk::StateMachine::CanHandleEvent(event);
  }
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectSeedTool(Action*, const StateEvent*)
{
  return m_Responder->SelectSeedTool();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectDrawTool(Action*, const StateEvent*)
{
  return m_Responder->SelectDrawTool();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::UnselectTools(Action*, const StateEvent*)
{
  return m_Responder->UnselectTools();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectPolyTool(Action*, const StateEvent*)
{
  return m_Responder->SelectPolyTool();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectViewMode(Action*, const StateEvent*)
{
  return m_Responder->SelectViewMode();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::CleanSlice(Action*, const StateEvent*)
{
  return m_Responder->CleanSlice();
}

} // end namespace
