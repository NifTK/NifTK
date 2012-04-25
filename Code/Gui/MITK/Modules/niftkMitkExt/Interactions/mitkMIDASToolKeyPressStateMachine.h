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
#ifndef MIDASTOOLKEYPRESSSTATEMACHINE
#define MIDASTOOLKEYPRESSSTATEMACHINE

#include "niftkMitkExtExports.h"
#include "mitkStateMachine.h"
#include "mitkMIDASToolKeyPressResponder.h"

namespace mitk {

/**
 * \class MIDASToolPressStateMachine
 * \brief StateMachine to check for key press events that MIDAS is interested in,
 * and pass them onto any registered MIDASToolKeyPressResponder.
 * \sa MIDASViewKeyPressResponder
 */
class NIFTKMITKEXT_EXPORT MIDASToolKeyPressStateMachine : public StateMachine
{

public:
  mitkClassMacro(MIDASToolKeyPressStateMachine, StateMachine); // this creates the Self typedef
  mitkNewMacro2Param(Self, const char*, MIDASToolKeyPressResponder*);

protected:

  /// \brief Purposely hidden, protected constructor, so class is instantiated via static ::New() macro, where we pass in the state machine pattern name, and also the object to pass the data onto.
  MIDASToolKeyPressStateMachine(const char * stateMachinePattern, MIDASToolKeyPressResponder* responder);

  /// \brief Purposely hidden, destructor.
  ~MIDASToolKeyPressStateMachine(){}

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
