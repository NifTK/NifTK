/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASToolKeyPressStateMachine.h"
#include "mitkWheelEvent.h"
#include "mitkStateEvent.h"

namespace mitk
{

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

bool MIDASToolKeyPressStateMachine::SelectSeedTool(Action*, const StateEvent*)
{
  return m_Responder->SelectSeedTool();
}

bool MIDASToolKeyPressStateMachine::SelectDrawTool(Action*, const StateEvent*)
{
  return m_Responder->SelectDrawTool();
}

bool MIDASToolKeyPressStateMachine::UnselectTools(Action*, const StateEvent*)
{
  return m_Responder->UnselectTools();
}

bool MIDASToolKeyPressStateMachine::SelectPolyTool(Action*, const StateEvent*)
{
  return m_Responder->SelectPolyTool();
}

bool MIDASToolKeyPressStateMachine::SelectViewMode(Action*, const StateEvent*)
{
  return m_Responder->SelectViewMode();
}

bool MIDASToolKeyPressStateMachine::CleanSlice(Action*, const StateEvent*)
{
  return m_Responder->CleanSlice();
}

} // end namespace
