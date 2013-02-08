/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "uk_ac_ucl_cmic_mitksegmentation_Activator.h"

#include <QtPlugin>

#include "MITKSegmentationView.h"
#include "QmitkCreatePolygonModelAction.h"
#include "QmitkSegmentationPreferencePage.h"

namespace mitk {

void uk_ac_ucl_cmic_mitksegmentation_Activator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(MITKSegmentationView, context)
  BERRY_REGISTER_EXTENSION_CLASS(QmitkCreatePolygonModelAction, context)
  BERRY_REGISTER_EXTENSION_CLASS(QmitkSegmentationPreferencePage, context)
}

void uk_ac_ucl_cmic_mitksegmentation_Activator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_mitksegmentation, mitk::uk_ac_ucl_cmic_mitksegmentation_Activator)
