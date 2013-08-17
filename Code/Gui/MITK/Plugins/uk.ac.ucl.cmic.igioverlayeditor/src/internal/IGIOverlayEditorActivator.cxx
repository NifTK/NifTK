/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIOverlayEditorActivator.h"

#include "../IGIOverlayEditor.h"
#include "IGIOverlayEditorPreferencePage.h"

namespace mitk {

ctkPluginContext* IGIOverlayEditorActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void IGIOverlayEditorActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(IGIOverlayEditor, context)
  BERRY_REGISTER_EXTENSION_CLASS(IGIOverlayEditorPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* IGIOverlayEditorActivator::getContext()
{
  return m_PluginContext;
}

//-----------------------------------------------------------------------------
} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igioverlayeditor, mitk::IGIOverlayEditorActivator)
