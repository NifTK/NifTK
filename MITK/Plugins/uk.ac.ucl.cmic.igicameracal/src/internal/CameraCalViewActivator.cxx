/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "CameraCalViewActivator.h"
#include <QtPlugin>
#include "CameraCalView.h"
#include "CameraCalViewPreferencePage.h"

namespace niftk
{

ctkPluginContext* CameraCalViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void CameraCalViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(CameraCalView, context)
  BERRY_REGISTER_EXTENSION_CLASS(CameraCalViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void CameraCalViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* CameraCalViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igicameracal, niftk::CameraCalViewActivator)
#endif
