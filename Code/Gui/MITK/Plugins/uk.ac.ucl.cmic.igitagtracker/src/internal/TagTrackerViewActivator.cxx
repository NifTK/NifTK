/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "TagTrackerViewActivator.h"
#include <QtPlugin>
#include "TagTrackerView.h"
#include "TagTrackerViewPreferencePage.h"

namespace mitk {

ctkPluginContext* TagTrackerViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void TagTrackerViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(TagTrackerView, context)
  BERRY_REGISTER_EXTENSION_CLASS(TagTrackerViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void TagTrackerViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* TagTrackerViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igitagtracker, mitk::TagTrackerViewActivator)
