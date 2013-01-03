/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center, 
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR 
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

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
