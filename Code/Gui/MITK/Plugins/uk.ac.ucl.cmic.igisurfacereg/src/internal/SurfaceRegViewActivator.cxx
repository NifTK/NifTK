/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SurfaceRegViewActivator.h"
#include <QtPlugin>
#include "SurfaceRegView.h"
#include "SurfaceRegViewPreferencePage.h"

namespace mitk {

ctkPluginContext* SurfaceRegViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void SurfaceRegViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(SurfaceRegView, context)
  BERRY_REGISTER_EXTENSION_CLASS(SurfaceRegViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void SurfaceRegViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* SurfaceRegViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igisurfacereg, mitk::SurfaceRegViewActivator)
