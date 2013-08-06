/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef AffineTransformViewActivator_h
#define AffineTransformViewActivator_h

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class AffineTransformViewActivator
 * \brief CTK Plugin Activator class for AffineTransformView.
 * \ingroup uk_ac_ucl_cmic_affinetransform_internal
 */
class AffineTransformViewActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

}; // AffineTransformViewActivator

}

#endif // AFFINETRANSFORMVIEWACTIVATOR_H
