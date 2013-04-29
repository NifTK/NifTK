/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "TrackedImageViewActivator.h"
#include <QtPlugin>
#include "TrackedImageView.h"
#include "TrackedImageViewPreferencePage.h"

namespace mitk {

ctkPluginContext* TrackedImageViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void TrackedImageViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(TrackedImageView, context)
  BERRY_REGISTER_EXTENSION_CLASS(TrackedImageViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void TrackedImageViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* TrackedImageViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igiTrackedImage, mitk::TrackedImageViewActivator)
