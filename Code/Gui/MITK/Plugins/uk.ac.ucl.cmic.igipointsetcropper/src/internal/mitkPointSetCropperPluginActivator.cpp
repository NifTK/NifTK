/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPointSetCropperPluginActivator.h"
#include "QmitkPointSetCropper.h"

#include <QtPlugin>

namespace mitk {

//-----------------------------------------------------------------------------
void PointSetCropperPluginActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS( QmitkPointSetCropper, context )
}


//-----------------------------------------------------------------------------
void PointSetCropperPluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

} // end namespace
Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_PointSetCropper, mitk::PointSetCropperPluginActivator)
