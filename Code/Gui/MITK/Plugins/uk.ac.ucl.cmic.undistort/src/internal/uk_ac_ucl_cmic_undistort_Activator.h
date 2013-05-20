/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef uk_ac_ucl_cmic_undistort_Activator_h
#define uk_ac_ucl_cmic_undistort_Activator_h

#include <ctkPluginActivator.h>

namespace mitk {

class uk_ac_ucl_cmic_undistort_Activator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

}; // uk_ac_ucl_cmic_undistort_Activator

}

#endif // uk_ac_ucl_cmic_undistort_Activator_h
