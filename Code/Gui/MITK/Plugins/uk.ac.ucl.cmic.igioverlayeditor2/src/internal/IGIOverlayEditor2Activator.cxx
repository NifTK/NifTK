/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIOverlayEditor2Activator.h"

#include "../IGIOverlayEditor2.h"
#include "IGIOverlayEditor2PreferencePage.h"

namespace mitk {

ctkPluginContext* IGIOverlayEditor2Activator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void IGIOverlayEditor2Activator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(IGIOverlayEditor2, context)
  BERRY_REGISTER_EXTENSION_CLASS(IGIOverlayEditor2PreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2Activator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* IGIOverlayEditor2Activator::getContext()
{
  return m_PluginContext;
}

//-----------------------------------------------------------------------------
} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igioverlayeditor2, mitk::IGIOverlayEditor2Activator)
