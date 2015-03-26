/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef IGIOverlayEditor2Activator_h
#define IGIOverlayEditor2Activator_h

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class IGIOverlayEditor2Activator
 * \brief Activator class for the IGIOverlayEditor2.
 * \ingroup uk_ac_ucl_cmic_igioverlayeditor2_internal
 */
class IGIOverlayEditor2Activator : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);
  static ctkPluginContext* getContext();

private:
  static ctkPluginContext* m_PluginContext;

};

} // end namespace

#endif /* IGIOverlayEditor2Activator_h */

