/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkDnDDisplayStateMachine_h
#define mitkDnDDisplayStateMachine_h

#include "niftkDnDDisplayExports.h"
#include "mitkDnDDisplayStateMachineResponder.h"
#include <mitkStateMachine.h>

#include <vector>

namespace mitk {

class BaseRenderer;

/**
 * \class DnDDisplayStateMachine
 * \brief StateMachine to check for key press events to switch window layout,
 * and pass them onto any registered DnDDisplaySMResponder.
 * \sa DnDDisplaySMResponder
 */
class NIFTKDNDDISPLAY_EXPORT DnDDisplayStateMachine : public StateMachine
{

public:
  mitkClassMacro(DnDDisplayStateMachine, StateMachine); // this creates the Self typedef
  mitkNewMacro2Param(Self, const char*, DnDDisplayStateMachineResponder*);

  // Currently, creating state machine using hard coded string, as I don't know where to load them from.
  static const std::string STATE_MACHINE_XML;

  /// \brief Tells if the state machine listens to the key events of the renderer.
  bool HasRenderer(const mitk::BaseRenderer* renderer) const;

  /// \brief Adds a renderer so that the state machine handles its key events.
  void AddRenderer(const mitk::BaseRenderer* renderer);

  /// \brief Removes a renderer so that the state machine does not handles its key events any more.
  void RemoveRenderer(const mitk::BaseRenderer* renderer);

protected:

  /// \brief Purposely hidden, protected constructor, so class is instantiated via static ::New() macro, where we pass in the state machine pattern name, and also the object to pass the data onto.
  DnDDisplayStateMachine(const char * stateMachinePattern, DnDDisplayStateMachineResponder* responder);

  /// \brief Purposely hidden, destructor.
  ~DnDDisplayStateMachine(){}

  virtual bool HandleEvent(StateEvent const* stateEvent);

  /// \see mitk::StateMachine::CanHandleEvent
  float CanHandleEvent(const StateEvent *) const;

  /// \brief Move in the anterior direction, simply passing method onto the DnDDisplayResponder
  bool MoveAnterior(Action*, const StateEvent*);

  /// \brief Move in the posterior direction, simply passing method onto the DnDDisplayResponder
  bool MovePosterior(Action*, const StateEvent*);

  /// \brief Switch the current view to Axial, simply passing method onto the DnDDisplayResponder
  bool SwitchToAxial(Action*, const StateEvent*);

  /// \brief Switch the current view to Sagittal, simply passing method onto the DnDDisplayResponder
  bool SwitchToSagittal(Action*, const StateEvent*);

  /// \brief Switch the current view to Coronal, simply passing method onto the DnDDisplayResponder
  bool SwitchToCoronal(Action*, const StateEvent*);

  /// \brief Switch the from single window to multiple windows or back
  bool ToggleMultiWindowLayout(Action*, const StateEvent*);

private:

  /// \brief the object that gets called, specified in constructor.
  DnDDisplayStateMachineResponder* m_Responder;

  /// \brief The renderers whose key events are listened to.
  std::vector<const mitk::BaseRenderer*> m_Renderers;

}; // end class

} // end namespace

#endif
