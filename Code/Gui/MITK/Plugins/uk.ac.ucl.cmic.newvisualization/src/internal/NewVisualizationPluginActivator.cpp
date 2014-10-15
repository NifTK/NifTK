/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#include "NewVisualizationPluginActivator.h"

#include <QtPlugin>

#include "NewVisualizationView.h"

namespace mitk {

NewVisualizationPluginActivator* NewVisualizationPluginActivator::s_Inst = 0;

NewVisualizationPluginActivator::NewVisualizationPluginActivator()
: m_Context(NULL)
{
  s_Inst = this;
}

NewVisualizationPluginActivator::~NewVisualizationPluginActivator()
{
}

void NewVisualizationPluginActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(NewVisualizationView, context)
  m_Context = context;

}

void NewVisualizationPluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_Context = 0;
}

NewVisualizationPluginActivator* NewVisualizationPluginActivator::GetDefault()
{
  return s_Inst;
}

ctkPluginContext* NewVisualizationPluginActivator::GetPluginContext() const
{
  return m_Context;
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_NewVisualization, mitk::NewVisualizationPluginActivator)
