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

  /// \brief Select the seed tool.
  virtual bool SelectSeedTool() = 0;

  /// \brief Select the draw tool.
  virtual bool SelectDrawTool() = 0;

  /// \brief Unselect the tool, (equivalent to selecting Posn in original MIDAS).
  virtual bool UnselectTools() = 0;

  /// \brief Select the poly tool.
  virtual bool SelectPolyTool() = 0;

  /// \brief Select the view mode.
  virtual bool SelectViewMode() = 0;

  /// \brief Clean the slice.
  virtual bool CleanSlice() = 0;

};

} // end namespace

#endif
