/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASSeedTool_h
#define mitkMIDASSeedTool_h

#include "niftkMIDASExports.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASPointSetInteractor.h"
#include <mitkStateEvent.h>

namespace mitk {

/**
 * \class MIDASSeedTool
 * \brief MIDAS seed tool for adding / removing / moving seeds.
 *
 * Interestingly, ANY of mitk::MIDASPolyTool, mitk::MIDASDrawTool or mitk::MIDASSeedTool can add seeds.
 * but only mitk::MIDASSeedTool can remove them.
 *
 * Provides
 * <pre>
 * 1. Right mouse button = place seed
 * 2. Left mouse button = select seed
 * 3. Move with left mouse button down = move selected seed
 * 4. Middle mouse button = select a seed if it is within a given distance and remove it.
 * </pre>
 * and includes Undo/Redo functionality. Given the above list, to remove seeds most people
 * hold the middle mouse button down, and drag it around, sucking up the seed points like a hoover.
 */
class NIFTKMIDAS_EXPORT MIDASSeedTool : public MIDASTool {

public:

  mitkClassMacro(MIDASSeedTool, MIDASTool);
  itkNewMacro(MIDASSeedTool);

  /// \see mitk::Tool::GetName()
  virtual const char* GetName() const;

  /// \see mitk::Tool::GetXPM()
  virtual const char** GetXPM() const;

  /// \brief When called, we create and register an mitkPointSetInteractor.
  virtual void Activated();

  /// \brief When called, we unregister the mitkPointSetInteractor.
  virtual void Deactivated();

  /// \brief Adds an event filter that can reject a state machine event or let it pass through.
  /// Overrides mitkMIDASStateMachine::InstallEventFilter() so that it adds every filter also to the
  /// internal point set interactor.
  virtual void InstallEventFilter(const MIDASEventFilter::Pointer eventFilter);

  /// \brief Removes an event filter that can reject a state machine event or let it pass through.
  /// Overrides mitkMIDASStateMachine::InstallEventFilter() to that it removes every filter also from the
  /// internal point set interactor.
  virtual void RemoveEventFilter(const MIDASEventFilter::Pointer eventFilter);

protected:

  MIDASSeedTool();
  virtual ~MIDASSeedTool();

  /// \see mitk::MIDASStateMachine::CanHandle
  virtual float CanHandle(const mitk::StateEvent* stateEvent) const;

private:

  mitk::MIDASPointSetInteractor::Pointer m_PointSetInteractor;

};

}

#endif
