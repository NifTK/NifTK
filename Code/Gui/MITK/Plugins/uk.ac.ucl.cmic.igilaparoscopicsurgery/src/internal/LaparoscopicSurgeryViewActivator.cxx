/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "LaparoscopicSurgeryViewActivator.h"
#include <QtPlugin>
#include "LaparoscopicSurgeryView.h"
#include "LaparoscopicSurgeryViewPreferencePage.h"

namespace mitk {

ctkPluginContext* LaparoscopicSurgeryViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void LaparoscopicSurgeryViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(LaparoscopicSurgeryView, context)
  BERRY_REGISTER_EXTENSION_CLASS(LaparoscopicSurgeryViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void LaparoscopicSurgeryViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* LaparoscopicSurgeryViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igilaparoscopicsurgery, mitk::LaparoscopicSurgeryViewActivator)
