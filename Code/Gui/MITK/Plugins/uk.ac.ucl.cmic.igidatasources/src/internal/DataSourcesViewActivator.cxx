/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "DataSourcesViewActivator.h"
#include <QtPlugin>
#include "DataSourcesView.h"
#include "DataSourcesViewPreferencePage.h"

namespace mitk {

ctkPluginContext* DataSourcesViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void DataSourcesViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(DataSourcesView, context)
  BERRY_REGISTER_EXTENSION_CLASS(DataSourcesViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void DataSourcesViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* DataSourcesViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igidatasources, mitk::DataSourcesViewActivator)
