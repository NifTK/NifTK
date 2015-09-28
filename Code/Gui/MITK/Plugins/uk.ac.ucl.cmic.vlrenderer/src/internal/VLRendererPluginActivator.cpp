/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "VLRendererPluginActivator.h"
#include <QtPlugin>
#include "VLRendererView.h"

namespace mitk {

VLRendererPluginActivator* VLRendererPluginActivator::s_Inst = 0;

VLRendererPluginActivator::VLRendererPluginActivator()
: m_Context(NULL)
{
  s_Inst = this;
}

VLRendererPluginActivator::~VLRendererPluginActivator()
{
}

void VLRendererPluginActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(VLRendererView, context)
  m_Context = context;

}

void VLRendererPluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_Context = 0;
}

VLRendererPluginActivator* VLRendererPluginActivator::GetDefault()
{
  return s_Inst;
}

ctkPluginContext* VLRendererPluginActivator::GetPluginContext() const
{
  return m_Context;
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_vlrenderer, mitk::VLRendererPluginActivator)
