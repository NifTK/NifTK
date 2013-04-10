/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "CommonLegacyActivator.h"
#include <QtPlugin>

namespace mitk {

ctkPluginContext* CommonLegacyActivator::s_PluginContext(NULL);

//-----------------------------------------------------------------------------
void CommonLegacyActivator::start(ctkPluginContext* context)
{
  s_PluginContext = context;
}


//-----------------------------------------------------------------------------
void CommonLegacyActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
ctkPluginContext* CommonLegacyActivator::GetPluginContext()
{
  return s_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_commonlegacy, mitk::CommonLegacyActivator)
