/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASVIEWKEYPRESSRESPONDER
#define MIDASVIEWKEYPRESSRESPONDER

#include "niftkMIDASExports.h"

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
class NIFTKMIDAS_EXPORT MIDASViewKeyPressResponder
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

  /// \brief Switch window layout.
  virtual bool SwitchWindowLayout() = 0;
};

} // end namespace

#endif
