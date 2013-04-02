/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "uk_ac_ucl_cmic_imagestatistics_Activator.h"

#include <QtPlugin>

#include "ImageStatisticsView.h"
#include "ImageStatisticsViewPreferencesPage.h"

namespace mitk {

void uk_ac_ucl_cmic_imagestatistics_Activator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(ImageStatisticsView, context)
  BERRY_REGISTER_EXTENSION_CLASS(ImageStatisticsViewPreferencesPage, context);
}

void uk_ac_ucl_cmic_imagestatistics_Activator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_imagestatistics, mitk::uk_ac_ucl_cmic_imagestatistics_Activator)
