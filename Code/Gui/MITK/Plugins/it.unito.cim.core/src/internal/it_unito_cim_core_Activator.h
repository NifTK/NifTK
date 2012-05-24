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

#ifndef it_unito_cim_core_Activator_h
#define it_unito_cim_core_Activator_h

#include <ctkPluginActivator.h>

class ImageInfoRenderer;

namespace mitk {

class it_unito_cim_core_ActivatorPrivate;

class it_unito_cim_core_Activator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:
  explicit it_unito_cim_core_Activator();
  virtual ~it_unito_cim_core_Activator();

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

private:
  void registerNodeDescriptors();

  QScopedPointer<it_unito_cim_core_ActivatorPrivate> d_ptr;

  Q_DECLARE_PRIVATE(it_unito_cim_core_Activator);
  Q_DISABLE_COPY(it_unito_cim_core_Activator);
}; // it_unito_cim_core_Activator

}

#endif // it_unito_cim_core_Activator_h
