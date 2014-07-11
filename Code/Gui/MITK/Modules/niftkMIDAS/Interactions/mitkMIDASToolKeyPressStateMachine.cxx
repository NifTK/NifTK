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

namespace mitk
{

//-----------------------------------------------------------------------------
MIDASToolKeyPressStateMachine::MIDASToolKeyPressStateMachine(MIDASToolKeyPressResponder* responder)
: mitk::EventStateMachine()
{
  assert(responder);
  m_Responder = responder;
}


//-----------------------------------------------------------------------------
MIDASToolKeyPressStateMachine::~MIDASToolKeyPressStateMachine()
{
}


//-----------------------------------------------------------------------------
void mitk::MIDASToolKeyPressStateMachine::ConnectActionsAndFunctions()
{
  CONNECT_FUNCTION("selectSeedTool", SelectSeedTool);
  CONNECT_FUNCTION("selectDrawTool", SelectDrawTool);
  CONNECT_FUNCTION("unselectTools", UnselectTools);
  CONNECT_FUNCTION("selectPolyTool", SelectPolyTool);
  CONNECT_FUNCTION("selectViewMode", SelectViewMode);
  CONNECT_FUNCTION("cleanSlice", CleanSlice);
}


//-----------------------------------------------------------------------------
bool mitk::MIDASToolKeyPressStateMachine::FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode)
{
  return this->CanHandleEvent(event);
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectSeedTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectSeedTool();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectDrawTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectDrawTool();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::UnselectTools(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->UnselectTools();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectPolyTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectPolyTool();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectViewMode(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectViewMode();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::CleanSlice(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->CleanSlice();
}

}
