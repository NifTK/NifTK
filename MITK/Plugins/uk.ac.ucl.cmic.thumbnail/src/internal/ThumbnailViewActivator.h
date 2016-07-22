/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ThumbnailViewActivator_h
#define ThumbnailViewActivator_h

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class ThumbnailViewActivator
 * \brief CTK Plugin Activator class for ThumbnailView.
 * \ingroup uk.ac.ucl.cmic.thumbnail_internal
 */
class ThumbnailViewActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
  Q_PLUGIN_METADATA(IID "uk_ac_ucl_cmic_thumbnail")
#endif

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

}; // ThumbnailViewActivator

}

#endif // THUMBNAILVIEWACTIVATOR_H
