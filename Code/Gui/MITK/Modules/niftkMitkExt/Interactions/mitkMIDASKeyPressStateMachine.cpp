/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-29 23:41:22 +0100 (Fri, 29 Jul 2011) $
 Revision          : $Revision: 6892 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASKeyPressStateMachine.h"
#include "mitkWheelEvent.h"
#include "mitkStateEvent.h"

namespace mitk
{

MIDASKeyPressStateMachine::MIDASKeyPressStateMachine(const char * stateMachinePattern, MIDASKeyPressResponder* responder)
: StateMachine(stateMachinePattern)
{
  assert(responder);
  m_Responder = responder;

  CONNECT_ACTION( 350001, MoveAnterior );
  CONNECT_ACTION( 350002, MovePosterior );
  CONNECT_ACTION( 350003, SwitchToAxial );
  CONNECT_ACTION( 350004, SwitchToSagittal );
  CONNECT_ACTION( 350005, SwitchToCoronal );
  CONNECT_ACTION( 350006, ScrollMouse );
}

bool MIDASKeyPressStateMachine::MoveAnterior(Action*, const StateEvent*)
{
  return m_Responder->MoveAnterior();
}

bool MIDASKeyPressStateMachine::MovePosterior(Action*, const StateEvent*)
{
  return m_Responder->MovePosterior();
}

bool MIDASKeyPressStateMachine::SwitchToAxial(Action*, const StateEvent*)
{
  return m_Responder->SwitchToAxial();
}

bool MIDASKeyPressStateMachine::SwitchToSagittal(Action*, const StateEvent*)
{
  return m_Responder->SwitchToSagittal();
}

bool MIDASKeyPressStateMachine::SwitchToCoronal(Action*, const StateEvent*)
{
  return m_Responder->SwitchToCoronal();
}

bool MIDASKeyPressStateMachine::ScrollMouse(Action*, const StateEvent* stateEvent)
{
  const WheelEvent* wheelEvent=dynamic_cast<const WheelEvent*>(stateEvent->GetEvent());
  if (wheelEvent != NULL)
  {
    int delta = wheelEvent->GetDelta();
    if ( delta < 0 )
    {
      return m_Responder->MovePosterior();
    }
    else
    {
      return m_Responder->MoveAnterior();
    }
  }
  return false;
}

} // end namespace
