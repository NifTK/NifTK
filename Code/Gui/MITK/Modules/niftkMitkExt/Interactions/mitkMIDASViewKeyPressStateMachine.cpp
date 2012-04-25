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

#include "mitkMIDASViewKeyPressStateMachine.h"
#include "mitkWheelEvent.h"
#include "mitkStateEvent.h"

namespace mitk
{

MIDASViewKeyPressStateMachine::MIDASViewKeyPressStateMachine(const char * stateMachinePattern, MIDASViewKeyPressResponder* responder)
: StateMachine(stateMachinePattern)
{
  assert(responder);
  m_Responder = responder;

  CONNECT_ACTION( 350001, MoveAnterior );
  CONNECT_ACTION( 350002, MovePosterior );
  CONNECT_ACTION( 350003, SwitchToAxial );
  CONNECT_ACTION( 350004, SwitchToSagittal );
  CONNECT_ACTION( 350005, SwitchToCoronal );
}

bool MIDASViewKeyPressStateMachine::MoveAnterior(Action*, const StateEvent*)
{
  return m_Responder->MoveAnterior();
}

bool MIDASViewKeyPressStateMachine::MovePosterior(Action*, const StateEvent*)
{
  return m_Responder->MovePosterior();
}

bool MIDASViewKeyPressStateMachine::SwitchToAxial(Action*, const StateEvent*)
{
  return m_Responder->SwitchToAxial();
}

bool MIDASViewKeyPressStateMachine::SwitchToSagittal(Action*, const StateEvent*)
{
  return m_Responder->SwitchToSagittal();
}

bool MIDASViewKeyPressStateMachine::SwitchToCoronal(Action*, const StateEvent*)
{
  return m_Responder->SwitchToCoronal();
}

} // end namespace
