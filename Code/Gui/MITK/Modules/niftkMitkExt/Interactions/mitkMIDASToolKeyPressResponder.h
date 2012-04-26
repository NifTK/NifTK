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
#ifndef MIDASTOOLKEYPRESSRESPONDER
#define MIDASTOOLKEYPRESSRESPONDER

#include "niftkMitkExtExports.h"

namespace mitk {


/**
 * \class MIDASToolKeyPressResponder
 * \brief Pure Virtual Interface to be implemented by classes that want to
 * respond to MIDAS key events relevant to which tool is selected where events
 * are passed through from the MIDASKeyPressStateMachine which is a subclass of
 * StateMachine, and hence registered with the interaction loop.
 *
 * \sa MIDASKeyPressStateMachine
 * \sa MIDASViewKeyPressResponder
 * \sa StateMachine
 */
class NIFTKMITKEXT_EXPORT MIDASToolKeyPressResponder
{
public:

  MIDASToolKeyPressResponder() {}
  virtual ~MIDASToolKeyPressResponder() {}

  /// \brief Select the seed tool, where in MIDAS this is the S key.
  virtual bool SelectSeedTool() = 0;

  /// \brief Select the draw tool, where in MIDAS this is the D key.
  virtual bool SelectDrawTool() = 0;

  /// \brief Unselect all tools, where in MIDAS is equivalent to selecting the Posn tool, which is Space or N.
  virtual bool UnselectTools() = 0;

  /// \brief Select the poly tool, where in MIDAS this is the Y key.
  virtual bool SelectPolyTool() = 0;

  /// \brief Select the view mode, where in  MIDAS this is the V key.
  virtual bool SelectViewMode() = 0;

  /// \brief Clean the slice, where in MIDAS this is the C key.
  virtual bool CleanSlice() = 0;

};

} // end namespace

#endif
