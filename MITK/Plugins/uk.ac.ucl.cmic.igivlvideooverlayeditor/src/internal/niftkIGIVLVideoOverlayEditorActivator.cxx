/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIVLVideoOverlayEditorActivator.h"
#include "niftkIGIVLVideoOverlayEditorPreferencePage.h"
#include "niftkIGIVLVideoOverlayEditor.h"

namespace niftk
{

ctkPluginContext* IGIVLVideoOverlayEditorActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditorActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(IGIVLVideoOverlayEditorPreferencePage, context)
  BERRY_REGISTER_EXTENSION_CLASS(IGIVLVideoOverlayEditor, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditorActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* IGIVLVideoOverlayEditorActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igivlvideooverlayeditor, niftk::IGIVLVideoOverlayEditorActivator)
#endif
