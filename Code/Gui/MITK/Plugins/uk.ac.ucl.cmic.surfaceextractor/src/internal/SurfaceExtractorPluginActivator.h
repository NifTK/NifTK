/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef SurfaceExtractorPluginActivator_h
#define SurfaceExtractorPluginActivator_h

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class SurfaceExtractorPluginActivator
 * \brief CTK Plugin Activator class for the Surface Extractor plugin.
 * \ingroup uk_ac_ucl_cmic_surfaceextractor_internal
 */
class SurfaceExtractorPluginActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

}; // SurfaceExtractorPluginActivator

}

#endif // SurfaceExtractorPluginActivator_h
