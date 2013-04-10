/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "NiftySegPluginActivator.h"

#include <QtPlugin>

#include "QmitkNiftySegView.h"

namespace mitk {

void NiftySegPluginActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftySegView, context)
}

void NiftySegPluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_niftyseg, mitk::NiftySegPluginActivator)
