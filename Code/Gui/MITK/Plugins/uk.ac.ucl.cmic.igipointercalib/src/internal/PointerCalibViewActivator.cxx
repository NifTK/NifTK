/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "PointerCalibViewActivator.h"
#include <QtPlugin>
#include "PointerCalibView.h"
#include "PointerCalibViewPreferencePage.h"

namespace mitk {

ctkPluginContext* PointerCalibViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void PointerCalibViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(PointerCalibView, context)
  BERRY_REGISTER_EXTENSION_CLASS(PointerCalibViewPreferencePage, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void PointerCalibViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* PointerCalibViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igipointercalib, mitk::PointerCalibViewActivator)
