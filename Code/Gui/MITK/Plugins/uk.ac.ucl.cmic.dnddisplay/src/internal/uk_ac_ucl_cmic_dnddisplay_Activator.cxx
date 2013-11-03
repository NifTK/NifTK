/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "uk_ac_ucl_cmic_dnddisplay_Activator.h"
#include <QmitkMIDASMultiViewEditor.h>
#include <QmitkMIDASMultiViewEditorPreferencePage.h>
#include <QtPlugin>
#include <mitkGlobalInteraction.h>
#include <Interactions/mitkDnDDisplayStateMachine.h>

namespace mitk {

ctkPluginContext* uk_ac_ucl_cmic_dnddisplay_Activator::s_PluginContext(NULL);

//-----------------------------------------------------------------------------
void uk_ac_ucl_cmic_dnddisplay_Activator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(QmitkMIDASMultiViewEditor, context)
  BERRY_REGISTER_EXTENSION_CLASS(QmitkMIDASMultiViewEditorPreferencePage, context);
  s_PluginContext = context;

  mitk::GlobalInteraction* globalInteractor =  mitk::GlobalInteraction::GetInstance();
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::DnDDisplayStateMachine::STATE_MACHINE_XML);
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
