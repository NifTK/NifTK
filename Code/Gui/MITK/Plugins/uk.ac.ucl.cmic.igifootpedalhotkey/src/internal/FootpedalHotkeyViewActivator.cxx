/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "FootpedalHotkeyViewActivator.h"
#include <QtPlugin>
#include "FootpedalHotkeyView.h"


namespace mitk
{


//-----------------------------------------------------------------------------
ctkPluginContext* FootpedalHotkeyViewActivator::m_PluginContext = 0;


//-----------------------------------------------------------------------------
void FootpedalHotkeyViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(FootpedalHotkeyView, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void FootpedalHotkeyViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* FootpedalHotkeyViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace


Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igifootpedalhotkey, mitk::FootpedalHotkeyViewActivator)
