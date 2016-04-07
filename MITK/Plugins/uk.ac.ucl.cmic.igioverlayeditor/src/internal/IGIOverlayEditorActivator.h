/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef IGIOverlayEditorActivator_h
#define IGIOverlayEditorActivator_h

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class IGIOverlayEditorActivator
 * \brief Activator class for the IGIOverlayEditor.
 * \ingroup uk_ac_ucl_cmic_igioverlayeditor_internal
 */
class IGIOverlayEditorActivator : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
  Q_PLUGIN_METADATA(IID "uk_ac_ucl_cmic_igioverlayeditor")
#endif

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);
  static ctkPluginContext* getContext();

private:
  static ctkPluginContext* m_PluginContext;

};

} // end namespace

#endif /* IGIOverlayEditorActivator_h */

