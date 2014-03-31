/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SideViewerActivator.h"

#include <QtPlugin>
#include <mitkGlobalInteraction.h>
#include <mitkMIDASTool.h>
#include "QmitkSideViewerView.h"

namespace mitk
{

ctkPluginContext* SideViewerActivator::s_PluginContext(NULL);

//-----------------------------------------------------------------------------
void SideViewerActivator::start(ctkPluginContext* context)
{
  s_PluginContext = context;
  BERRY_REGISTER_EXTENSION_CLASS(QmitkSideViewerView, context);

  mitk::MIDASTool::LoadBehaviourStrings();
}


//-----------------------------------------------------------------------------
void SideViewerActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
ctkPluginContext* SideViewerActivator::GetPluginContext()
{
  return s_PluginContext;
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_sideviewer, mitk::SideViewerActivator)
