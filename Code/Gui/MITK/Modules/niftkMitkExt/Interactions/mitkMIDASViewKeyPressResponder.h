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
#ifndef MIDASVIEWKEYPRESSRESPONDER
#define MIDASVIEWKEYPRESSRESPONDER

#include "niftkMitkExtExports.h"

namespace mitk {

/**
 * \class MIDASViewKeyPressResponder
 * \brief Pure Virtual Interface to be implemented by classes that want to
 * respond to MIDAS key events relevant to how the image is viewed where events
 * are passed through from the MIDASKeyPressStateMachine which is a subclass of
 * StateMachine, and hence registered with the interaction loop.
 *
 * \sa MIDASKeyPressStateMachine
 * \sa MIDASToolKeyPressResponder
 * \sa StateMachine
 */
class NIFTKMITKEXT_EXPORT MIDASViewKeyPressResponder
{
public:

  MIDASViewKeyPressResponder() {}
  virtual ~MIDASViewKeyPressResponder() {}

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

} // end namespace

#endif
