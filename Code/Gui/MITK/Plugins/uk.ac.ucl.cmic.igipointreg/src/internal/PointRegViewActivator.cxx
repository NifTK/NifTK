/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "PointRegViewActivator.h"
#include <QtPlugin>
#include "PointRegView.h"
#include "PointRegViewPreferencePage.h"

namespace mitk {

ctkPluginContext* PointRegViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void PointRegViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(PointRegView, context)
  BERRY_REGISTER_EXTENSION_CLASS(PointRegViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void PointRegViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* PointRegViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igipointreg, mitk::PointRegViewActivator)
