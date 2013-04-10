/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "uk_ac_ucl_cmic_surgicalguidance_Activator.h"
#include <QtPlugin>
#include "SurgicalGuidanceView.h"
#include "SurgicalGuidanceViewPreferencePage.h"

namespace mitk {

ctkPluginContext* uk_ac_ucl_cmic_surgicalguidance_Activator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void uk_ac_ucl_cmic_surgicalguidance_Activator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(SurgicalGuidanceView, context)
  BERRY_REGISTER_EXTENSION_CLASS(SurgicalGuidanceViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void uk_ac_ucl_cmic_surgicalguidance_Activator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* uk_ac_ucl_cmic_surgicalguidance_Activator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_surgicalguidance, mitk::uk_ac_ucl_cmic_surgicalguidance_Activator)
