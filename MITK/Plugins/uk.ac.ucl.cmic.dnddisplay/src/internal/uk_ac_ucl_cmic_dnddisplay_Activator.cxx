/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "uk_ac_ucl_cmic_dnddisplay_Activator.h"
#include <QmitkDnDDisplayPreferencePage.h>
#include <QmitkMultiViewerEditor.h>
//#include <QmitkSingleViewerEditor.h>
#include <QtPlugin>

namespace mitk
{

ctkPluginContext* uk_ac_ucl_cmic_dnddisplay_Activator::s_PluginContext(NULL);

//-----------------------------------------------------------------------------
void uk_ac_ucl_cmic_dnddisplay_Activator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(QmitkMultiViewerEditor, context);
//  BERRY_REGISTER_EXTENSION_CLASS(QmitkSingleViewerEditor, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkDnDDisplayPreferencePage, context);
  s_PluginContext = context;
}


//-----------------------------------------------------------------------------
void uk_ac_ucl_cmic_dnddisplay_Activator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
ctkPluginContext* uk_ac_ucl_cmic_dnddisplay_Activator::GetPluginContext()
{
  return s_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_dnddisplay, mitk::uk_ac_ucl_cmic_dnddisplay_Activator)
