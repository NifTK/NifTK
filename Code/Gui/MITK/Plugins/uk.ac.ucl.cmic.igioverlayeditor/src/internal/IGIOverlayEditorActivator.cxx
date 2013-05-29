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

//-----------------------------------------------------------------------------
void IGIOverlayEditorActivator::start(ctkPluginContext* context)
{
  Q_UNUSED(context)

  BERRY_REGISTER_EXTENSION_CLASS(IGIOverlayEditor, context)
  BERRY_REGISTER_EXTENSION_CLASS(IGIOverlayEditorPreferencePage, context)
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igioverlayeditor, mitk::IGIOverlayEditorActivator)
