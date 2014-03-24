/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SideViewActivator.h"

#include <QtPlugin>
#include <mitkGlobalInteraction.h>
#include <mitkMIDASTool.h>
#include "QmitkSideViewView.h"

namespace mitk
{

ctkPluginContext* SideViewActivator::s_PluginContext(NULL);

//-----------------------------------------------------------------------------
void SideViewActivator::start(ctkPluginContext* context)
{
  s_PluginContext = context;
  BERRY_REGISTER_EXTENSION_CLASS(QmitkSideViewView, context);

  mitk::MIDASTool::LoadBehaviourStrings();
}


//-----------------------------------------------------------------------------
void SideViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
ctkPluginContext* SideViewActivator::GetPluginContext()
{
  return s_PluginContext;
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_sideview, mitk::SideViewActivator)
