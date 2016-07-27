/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPluginActivator.h"

#include "niftkThumbnailViewPreferencePage.h"
#include "niftkThumbnailView.h"
#include <QtPlugin>

namespace niftk
{

//-----------------------------------------------------------------------------
void PluginActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(ThumbnailView, context);
  BERRY_REGISTER_EXTENSION_CLASS(ThumbnailViewPreferencePage, context);
}


//-----------------------------------------------------------------------------
void PluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_thumbnailview, niftk::PluginActivator)
#endif
