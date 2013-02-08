/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "uk_ac_ucl_cmic_singlewidgeteditor_Activator.h"

#include "../QmitkSingleWidgetEditor.h"
#include "QmitkSingleWidgetEditorPreferencePage.h"


void
uk_ac_ucl_cmic_singlewidgeteditor_Activator::start(ctkPluginContext* context)
{
  Q_UNUSED(context)

  BERRY_REGISTER_EXTENSION_CLASS(QmitkSingleWidgetEditor, context)
  BERRY_REGISTER_EXTENSION_CLASS(QmitkSingleWidgetEditorPreferencePage, context)
}

void
uk_ac_ucl_cmic_singlewidgeteditor_Activator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_singlewidgeteditor, uk_ac_ucl_cmic_singlewidgeteditor_Activator)
