/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SurfaceReconViewActivator.h"
#include <QtPlugin>
#include "SurfaceReconView.h"
#include "SurfaceReconViewPreferencePage.h"

namespace mitk {

ctkPluginContext* SurfaceReconViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void SurfaceReconViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(SurfaceReconView, context)
  BERRY_REGISTER_EXTENSION_CLASS(SurfaceReconViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void SurfaceReconViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* SurfaceReconViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igisurfacerecon, mitk::SurfaceReconViewActivator)
