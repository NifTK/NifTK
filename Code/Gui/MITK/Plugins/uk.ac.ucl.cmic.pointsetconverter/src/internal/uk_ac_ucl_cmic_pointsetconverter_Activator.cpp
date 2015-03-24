/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#include "uk_ac_ucl_cmic_pointsetconverter_Activator.h"

#include <QtPlugin>

#include "PointSetConverterView.h"

namespace mitk {

void uk_ac_ucl_cmic_pointsetconverter_Activator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(PointSetConverterView, context)
}

void uk_ac_ucl_cmic_pointsetconverter_Activator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_pointsetconverter, mitk::uk_ac_ucl_cmic_pointsetconverter_Activator)
#endif
