/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "CommonActivator.h"
#include <QtPlugin>

namespace mitk {

ctkPluginContext* CommonActivator::s_PluginContext(NULL);

void CommonActivator::start(ctkPluginContext* context)
{
  s_PluginContext = context;
}

void CommonActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

ctkPluginContext* CommonActivator::GetPluginContext()
{
  return s_PluginContext;
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_common, mitk::CommonActivator)
