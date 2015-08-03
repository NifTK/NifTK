/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "PivotCalibrationViewActivator.h"
#include <QtPlugin>
#include "PivotCalibrationView.h"

namespace mitk {

ctkPluginContext* PivotCalibrationViewActivator::m_PluginContext = 0;

//-----------------------------------------------------------------------------
void PivotCalibrationViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(PivotCalibrationView, context)
  m_PluginContext = context;
}


//-----------------------------------------------------------------------------
void PivotCalibrationViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_PluginContext = NULL;
}


//-----------------------------------------------------------------------------
ctkPluginContext* PivotCalibrationViewActivator::getContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igipivotcalibration, mitk::PivotCalibrationViewActivator)
