/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitk_MIDASToolKeyPressStateMachine_h
#define mitk_MIDASToolKeyPressStateMachine_h

#include "niftkMIDASExports.h"
#include "mitkMIDASToolKeyPressResponder.h"
#include <mitkStateMachine.h>

namespace mitk {

/**
 * \class MIDASToolPressStateMachine
 * \brief StateMachine to check for key press events that MIDAS is interested in,
 * and pass them onto any registered MIDASToolKeyPressResponder.
 * \sa MIDASViewKeyPressResponder
 */
class NIFTKMIDAS_EXPORT MIDASToolKeyPressStateMachine : public StateMachine
{

public:
  mitkClassMacro(MIDASToolKeyPressStateMachine, StateMachine); // this creates the Self typedef
  mitkNewMacro2Param(Self, const char*, MIDASToolKeyPressResponder*);

protected:

  /// \brief Purposely hidden, protected constructor, so class is instantiated via static ::New() macro, where we pass in the state machine pattern name, and also the object to pass the data onto.
  MIDASToolKeyPressStateMachine(const char * stateMachinePattern, MIDASToolKeyPressResponder* responder);

  /// \brief Purposely hidden, destructor.
  ~MIDASToolKeyPressStateMachine(){}

  /// \see mitk::StateMachine::CanHandleEvent
  float CanHandleEvent(const StateEvent *) const;

  /// \see mitk::MIDASToolKeyPressResponder::SelectSeedTool()
  bool SelectSeedTool(Action*, const StateEvent*);

  /// \see mitk::MIDASToolKeyPressResponder::SelectDrawTool()
  bool SelectDrawTool(Action*, const StateEvent*);

  /// \see mitk::MIDASToolKeyPressResponder::UnselectTools()
  bool UnselectTools(Action*, const StateEvent*);

  /// \see mitk::MIDASToolKeyPressResponder::SelectPolyTool()
  bool SelectPolyTool(Action*, const StateEvent*);

  /// \see mitk::MIDASToolKeyPressResponder::SelectViewMode()
  bool SelectViewMode(Action*, const StateEvent*);

  /// \see mitk::MIDASToolKeyPressResponder::CleanSlice()
  bool CleanSlice(Action*, const StateEvent*);

private:

  /// \brief the object that gets called, specified in constructor.
  MIDASToolKeyPressResponder* m_Responder;

}; // end class

} // end namespace

#endif
