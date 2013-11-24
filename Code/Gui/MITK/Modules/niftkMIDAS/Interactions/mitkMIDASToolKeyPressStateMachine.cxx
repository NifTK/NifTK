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
float MIDASToolKeyPressStateMachine::CanHandle(const mitk::StateEvent* stateEvent) const
{
  // See StateMachine.xml for event Ids.
  int eventId = stateEvent->GetId();
  if (eventId == 18   // S
      || eventId == 4004 // D
      || eventId == 4018 // Y
      || eventId == 4015 // V
      || eventId == 4003 // C
      || eventId == 13   // N
      || eventId == 25   // Space
      )
  {
    return 1.0f;
  }
  else
  {
    // Note that the superclass is not a MIDAS state machine and it does not
    // have a CanHandle function.
    return Superclass::CanHandleEvent(stateEvent);
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
