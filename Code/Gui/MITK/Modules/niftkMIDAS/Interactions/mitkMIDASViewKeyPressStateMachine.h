/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASVIEWKEYPRESSSTATEMACHINE
#define MIDASVIEWKEYPRESSSTATEMACHINE

#include "niftkMIDASExports.h"
#include "mitkMIDASViewKeyPressResponder.h"
#include <mitkStateMachine.h>

namespace mitk {

/**
 * \class MIDASViewPressStateMachine
 * \brief StateMachine to check for key press events that MIDAS is interested in,
 * and pass them onto any registered MIDASViewKeyPressResponder.
 * \sa MIDASViewKeyPressResponder
 */
class NIFTKMIDAS_EXPORT MIDASViewKeyPressStateMachine : public StateMachine
{

public:
  mitkClassMacro(MIDASViewKeyPressStateMachine, StateMachine); // this creates the Self typedef
  mitkNewMacro2Param(Self, const char*, MIDASViewKeyPressResponder*);

protected:

  /// \brief Purposely hidden, protected constructor, so class is instantiated via static ::New() macro, where we pass in the state machine pattern name, and also the object to pass the data onto.
  MIDASViewKeyPressStateMachine(const char * stateMachinePattern, MIDASViewKeyPressResponder* responder);

  /// \brief Purposely hidden, destructor.
  ~MIDASViewKeyPressStateMachine(){}

  /// \see mitk::StateMachine::CanHandleEvent
  float CanHandleEvent(const StateEvent *) const;

  /// \brief Move in the anterior direction, simply passing method onto the MIDASViewKeyPressResponder
  bool MoveAnterior(Action*, const StateEvent*);

  /// \brief Move in the posterior direction, simply passing method onto the MIDASViewKeyPressResponder
  bool MovePosterior(Action*, const StateEvent*);

  /// \brief Switch the current view to Axial, simply passing method onto the MIDASViewKeyPressResponder
  bool SwitchToAxial(Action*, const StateEvent*);

  /// \brief Switch the current view to Sagittal, simply passing method onto the MIDASViewKeyPressResponder
  bool SwitchToSagittal(Action*, const StateEvent*);

  /// \brief Switch the current view to Coronal, simply passing method onto the MIDASViewKeyPressResponder
  bool SwitchToCoronal(Action*, const StateEvent*);

private:

  /// \brief the object that gets called, specified in constructor.
  MIDASViewKeyPressResponder* m_Responder;

}; // end class

} // end namespace

#endif
