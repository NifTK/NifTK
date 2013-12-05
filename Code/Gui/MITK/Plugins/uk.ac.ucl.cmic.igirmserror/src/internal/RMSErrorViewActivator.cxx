/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "RMSErrorViewActivator.h"
#include "RMSErrorView.h"
#include <QtPlugin>

namespace mitk {

ctkPluginContext* RMSErrorViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void RMSErrorViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(RMSErrorView, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void RMSErrorViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* RMSErrorViewActivator::getContext()
{
  return m_PluginContext;
}


//-----------------------------------------------------------------------------
} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igirmserror, mitk::RMSErrorViewActivator)
