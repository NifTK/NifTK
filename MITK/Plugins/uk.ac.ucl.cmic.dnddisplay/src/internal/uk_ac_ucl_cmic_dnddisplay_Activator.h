/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef uk_ac_ucl_cmic_dnddisplay_Activator_h
#define uk_ac_ucl_cmic_dnddisplay_Activator_h

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class uk_ac_ucl_cmic_dnddisplay_Activator
 * \brief CTK Plugin Activator class for the DnD Display Plugin.
 * \ingroup uk_ac_ucl_cmic_dnddisplay_internal
 */
class uk_ac_ucl_cmic_dnddisplay_Activator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

  static ctkPluginContext* GetPluginContext();

private:

  static ctkPluginContext* s_PluginContext;

}; // uk_ac_ucl_cmic_dnddisplay_Activator

}

#endif
