/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDataSourcesViewActivator.h"
#include "niftkDataSourcesView.h"
#include "niftkDataSourcesViewPreferencePage.h"
#include <QtPlugin>

namespace niftk
{

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

#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igidatasources, niftk::DataSourcesViewActivator)
#endif
