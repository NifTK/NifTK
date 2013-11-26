/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASEventFilter_h
#define mitkMIDASEventFilter_h

#include "niftkMIDASExports.h"

#include <mitkCommon.h>

namespace mitk {

class StateEvent;

/**
 * \class MIDASEventFilter
 *
 * \brief MIDASEventFilter represents a condition that has to be fulfilled
 * so that an event is processed by a MIDAS state machine.
 *
 * This can be used e.g. to restrict the scope of a tool or interactor to specific
 * render windows.
 */
class NIFTKMIDAS_EXPORT MIDASEventFilter : public itk::Object
{

public:

  mitkClassMacro(MIDASEventFilter, itk::Object);

  /// \brief Returns true if the event should be filtered, i.e. not processed,
  /// otherwise false.
  virtual bool EventFilter(const mitk::StateEvent* stateEvent) const = 0;

protected:

  MIDASEventFilter(); // purposefully hidden
  virtual ~MIDASEventFilter(); // purposefully hidden

private:

};

}

#endif
