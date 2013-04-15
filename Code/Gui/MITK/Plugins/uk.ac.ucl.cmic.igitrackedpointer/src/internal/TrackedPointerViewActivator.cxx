/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "TrackedPointerViewActivator.h"
#include <QtPlugin>
#include "TrackedPointerView.h"
#include "TrackedPointerViewPreferencePage.h"

namespace mitk {

ctkPluginContext* TrackedPointerViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void TrackedPointerViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(TrackedPointerView, context)
  BERRY_REGISTER_EXTENSION_CLASS(TrackedPointerViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void TrackedPointerViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* TrackedPointerViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igitrackedpointer, mitk::TrackedPointerViewActivator)
