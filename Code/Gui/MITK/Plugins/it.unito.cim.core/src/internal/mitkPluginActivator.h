/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKPLUGINACTIVATOR_H
#define MITKPLUGINACTIVATOR_H

#include <ctkPluginActivator.h>

class ImageInfoRenderer;

namespace mitk {

class PluginActivatorPrivate;

class PluginActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:
  explicit PluginActivator();
  virtual ~PluginActivator();

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

private:
  void registerNodeDescriptors();

  QScopedPointer<PluginActivatorPrivate> d_ptr;

  Q_DECLARE_PRIVATE(PluginActivator);
  Q_DISABLE_COPY(PluginActivator);
}; // PluginActivator

}

#endif // MITKPLUGINACTIVATOR_H
