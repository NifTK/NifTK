/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ImageLookupTablesViewActivator_h
#define ImageLookupTablesViewActivator_h

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class ImageLookupTablesViewActivator
 * \brief CTK Plugin Activator class for ImageLookupTablesView.
 * \ingroup uk_ac_ucl_cmic_imagelookuptables_internal
 */
class ImageLookupTablesViewActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

}; // ImageLookupTablesViewActivator

} // end namespace

#endif
