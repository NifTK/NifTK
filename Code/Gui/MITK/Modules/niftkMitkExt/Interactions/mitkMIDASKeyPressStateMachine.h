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
#ifndef MIDASKEYPRESSSTATEMACHINE
#define MIDASKEYPRESSSTATEMACHINE

#include "niftkMitkExtExports.h"
#include "mitkStateMachine.h"

namespace mitk {

/**
 * \class MIDASKeyPressResponder
 * \brief Pure Virtual Interface to be implemented by classes that want to
 * respond to MIDAS key events passed through from the MIDASKeyPressStateMachine
 * which is a subclass of StateMachine, and hence registered with the interaction loop.
 *
 * \sa MIDASKeyPressStateMachine
 * \sa StateMachine
 */
class NIFTKMITKEXT_EXPORT MIDASKeyPressResponder
{
public:

  /// \brief Move anterior a slice.
  virtual bool MoveAnterior() = 0;

  /// \brief Move posterior a slice.
  virtual bool MovePosterior() = 0;

  /// \brief Switch to Axial.
  virtual bool SwitchToAxial() = 0;

  /// \brief Switch to Sagittal.
  virtual bool SwitchToSagittal() = 0;

  /// \brief Switch to Coronal.
  virtual bool SwitchToCoronal() = 0;
};

/**
 * \class MIDASKeyPressStateMachine
 * \brief StateMachine to check for key press events that MIDAS is interested in, and pass them onto a registered MIDASKeyPressResponder.
 * \sa MIDASKeyPressResponder
 */
class NIFTKMITKEXT_EXPORT MIDASKeyPressStateMachine : public StateMachine
{

public:
  mitkClassMacro(MIDASKeyPressStateMachine, StateMachine); // this creates the Self typedef
  mitkNewMacro2Param(Self, const char*, MIDASKeyPressResponder*);

protected:

  /// \brief Purposely hidden, protected constructor, so class is instantiated via static ::New() macro, where we pass in the state machine pattern name, and also the object to pass the data onto.
  MIDASKeyPressStateMachine(const char * stateMachinePattern, MIDASKeyPressResponder* responder);

  /// \brief Purposely hidden, destructor.
  ~MIDASKeyPressStateMachine(){}

  /// \brief Move in the anterior direction, simply passing method onto the MIDASKeyPressResponder
  bool MoveAnterior(Action*, const StateEvent*);

  /// \brief Move in the posterior direction, simply passing method onto the MIDASKeyPressResponder
  bool MovePosterior(Action*, const StateEvent*);

  /// \brief Switch the current view to Axial, simply passing method onto the MIDASKeyPressResponder
  bool SwitchToAxial(Action*, const StateEvent*);

  /// \brief Switch the current view to Sagittal, simply passing method onto the MIDASKeyPressResponder
  bool SwitchToSagittal(Action*, const StateEvent*);

  /// \brief Switch the current view to Coronal, simply passing method onto the MIDASKeyPressResponder
  bool SwitchToCoronal(Action*, const StateEvent*);

  /// \brief Respond to mouse wheel scroll events.
  bool ScrollMouse(Action*, const StateEvent*);

private:

  /// \brief the object that gets called, specified in constructor.
  MIDASKeyPressResponder* m_Responder;

}; // end class

} // end namespace

#endif
