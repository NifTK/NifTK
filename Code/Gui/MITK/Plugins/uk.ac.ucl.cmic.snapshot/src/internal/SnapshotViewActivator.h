/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef SnapshotViewActivator_h
#define SnapshotViewActivator_h

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class SnapshotViewActivator
 * \brief CTK Plugin Activator class for SnapshotView.
 * \ingroup uk_ac_ucl_cmic_snapshot_internal
 */
class SnapshotViewActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

}; // SnapshotViewActivator

}

#endif // SNAPSHOTVIEWACTIVATOR_H
