/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIVLEditorActivator.h"

#include "../IGIVLEditor.h"
#include "IGIVLEditorPreferencePage.h"

namespace mitk
{


//-----------------------------------------------------------------------------
ctkPluginContext*               IGIVLEditorActivator::s_PluginContext = 0;


//-----------------------------------------------------------------------------
void IGIVLEditorActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(IGIVLEditor, context)
  BERRY_REGISTER_EXTENSION_CLASS(IGIVLEditorPreferencePage, context)
  s_PluginContext = context;
}


//-----------------------------------------------------------------------------
void IGIVLEditorActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  s_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* IGIVLEditorActivator::getContext()
{
  return s_PluginContext;
}


//-----------------------------------------------------------------------------
} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igivleditor, mitk::IGIVLEditorActivator)
