/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef IGIVLEditorActivator_h
#define IGIVLEditorActivator_h

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class IGIVLEditorActivator
 * \brief Activator class for the IGIVLEditor.
 * \ingroup uk_ac_ucl_cmic_igivleditor_internal
 */
class IGIVLEditorActivator : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:
  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);
  static ctkPluginContext* getContext();

private:
  static ctkPluginContext*                s_PluginContext;
};

} // end namespace

#endif /* IGIVLEditorActivator_h */

