/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "MIDASActivator.h"
#include <QtPlugin>
#include <mitkGlobalInteraction.h>
#include <mitkMIDASTool.h>

namespace mitk {

ctkPluginContext* MIDASActivator::s_PluginContext(NULL);

//-----------------------------------------------------------------------------
void MIDASActivator::start(ctkPluginContext* context)
{
  s_PluginContext = context;

  mitk::GlobalInteraction* globalInteractor =  mitk::GlobalInteraction::GetInstance();

  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_SEED_DROPPER_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_SEED_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_DRAW_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_POLY_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_TOOL_KEYPRESS_STATE_MACHINE_XML);
}


//-----------------------------------------------------------------------------
void MIDASActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
ctkPluginContext* MIDASActivator::GetPluginContext()
{
  return s_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_commonmidas, mitk::MIDASActivator)
