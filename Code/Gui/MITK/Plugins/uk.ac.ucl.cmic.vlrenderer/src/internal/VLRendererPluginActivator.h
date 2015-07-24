/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#ifndef VLRendererPluginActivator_h
#define VLRendererPluginActivator_h

#include <ctkPluginActivator.h>

namespace mitk {

class VLRendererPluginActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:
  VLRendererPluginActivator();
  ~VLRendererPluginActivator();

  static VLRendererPluginActivator* GetDefault();  
  ctkPluginContext* GetPluginContext() const;

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

private:
  static VLRendererPluginActivator* s_Inst;
  ctkPluginContext* m_Context;
}; 

}

#endif // VLRendererPluginActivator_h
