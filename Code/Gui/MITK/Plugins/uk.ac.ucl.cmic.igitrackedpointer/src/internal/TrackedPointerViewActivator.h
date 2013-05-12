/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef TrackedPointerViewActivator_h
#define TrackedPointerViewActivator_h

#include <ctkPluginActivator.h>

namespace mitk {

class TrackedPointerViewActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);
  static ctkPluginContext* getContext();

private:
  static ctkPluginContext* m_PluginContext;

}; // end class

} // end namespace

#endif // TrackedPointerViewActivator_h
