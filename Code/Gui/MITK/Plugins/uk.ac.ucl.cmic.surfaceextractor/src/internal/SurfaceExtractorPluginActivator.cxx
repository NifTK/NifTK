/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SurfaceExtractorPluginActivator.h"

#include "SurfaceExtractorView.h"
#include "SurfaceExtractorPreferencePage.h"

#include <QtPlugin>

namespace mitk {

void SurfaceExtractorPluginActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(SurfaceExtractorView, context);
  BERRY_REGISTER_EXTENSION_CLASS(SurfaceExtractorPreferencePage, context);
}

void SurfaceExtractorPluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_surfaceextractor, mitk::SurfaceExtractorPluginActivator)
