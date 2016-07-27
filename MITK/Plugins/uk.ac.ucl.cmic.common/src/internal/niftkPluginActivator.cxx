/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPluginActivator.h"

#include <QtPlugin>

namespace niftk
{

PluginActivator* PluginActivator::s_Instance = nullptr;

//-----------------------------------------------------------------------------
PluginActivator::PluginActivator()
{
  s_Instance = this;
}


//-----------------------------------------------------------------------------
PluginActivator::~PluginActivator()
{
}


//-----------------------------------------------------------------------------
PluginActivator* PluginActivator::GetInstance()
{
  return s_Instance;
}


//-----------------------------------------------------------------------------
ctkPluginContext* PluginActivator::GetContext() const
{
  return m_Context;
}


//-----------------------------------------------------------------------------
void PluginActivator::start(ctkPluginContext* context)
{
  m_Context = context;
}


//-----------------------------------------------------------------------------
void PluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context);
}

}

#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_common, niftk::PluginActivator)
#endif
