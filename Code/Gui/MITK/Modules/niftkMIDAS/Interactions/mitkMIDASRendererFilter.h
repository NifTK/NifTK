/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASRendererFilter_h
#define mitkMIDASRendererFilter_h

#include "niftkMIDASExports.h"

#include "mitkMIDASEventFilter.h"

#include <mitkCommon.h>
#include <vector>

namespace mitk {

class BaseRenderer;

/**
 * \class MIDASRendererFilter
 *
 * \brief MIDASRendererFilter represents a condition that allows only the events
 * coming from certain renderers to be processed.
 *
 * Events from other renderers will be rejected.
 */
class NIFTKMIDAS_EXPORT MIDASRendererFilter : public MIDASEventFilter
{

public:

  mitkClassMacro(MIDASRendererFilter, MIDASEventFilter);

  /// \brief Returns true if the sender of the event (the renderer where the event
  /// comes from) is not among the renderers added to this object.
  /// Otherwise, it returns false.
  virtual bool EventFilter(const mitk::StateEvent* stateEvent) const;

  /// \brief Adds the renderer to the list of allowed event sources.
  void AddRenderer(mitk::BaseRenderer* renderer);

  /// \brief Removes the renderer from the list of allowed event sources.
  void RemoveRenderer(mitk::BaseRenderer* renderer);

protected:

  MIDASRendererFilter(); // purposefully hidden
  virtual ~MIDASRendererFilter(); // purposefully hidden

private:

  /// \brief The list of allowed event sources.
  std::vector<mitk::BaseRenderer*> m_Renderers;

};

}

#endif
