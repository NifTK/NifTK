/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#ifndef NewVisualizationPluginActivator_h
#define NewVisualizationPluginActivator_h

#include <ctkPluginActivator.h>
#include <mitkOCLResourceService.h>

namespace mitk {

class NewVisualizationPluginActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:
    NewVisualizationPluginActivator();
  ~NewVisualizationPluginActivator();

  static NewVisualizationPluginActivator* GetDefault();  
  ctkPluginContext* GetPluginContext() const;

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

  static OclResourceService * GetOpenCLService();

private:
  static NewVisualizationPluginActivator* s_Inst;
  ctkPluginContext* m_Context;
}; // RiskVisulaizationPluginActivator

}

#endif // NewVisualizationPluginActivator_h
