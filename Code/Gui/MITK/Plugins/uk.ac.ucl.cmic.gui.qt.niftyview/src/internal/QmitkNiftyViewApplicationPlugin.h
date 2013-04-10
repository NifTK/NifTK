/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKNIFTYVIEWAPPLICATIONPLUGIN_H_
#define QMITKNIFTYVIEWAPPLICATIONPLUGIN_H_

#include <berryAbstractUICTKPlugin.h>
#include "QmitkCommonAppsApplicationPlugin.h"

/**
 * \class QmitkNiftyViewApplicationPlugin
 * \brief Implements QT and CTK specific functionality to launch the application as a plugin.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview_internal
 */
class QmitkNiftyViewApplicationPlugin : public QmitkCommonAppsApplicationPlugin, public berry::AbstractUICTKPlugin
{
  Q_OBJECT
  
public:

  QmitkNiftyViewApplicationPlugin();
  ~QmitkNiftyViewApplicationPlugin();

  void start(ctkPluginContext*);
  void stop(ctkPluginContext*);

protected:

  /// \brief Called each time a data node is added, so we make sure it is initialised with a Window/Level.
  virtual void NodeAdded(const mitk::DataNode *node);

  /// \brief Called by framework to get a URL for help system.
  virtual QString GetHelpHomePageURL() const;

private:

};

#endif /* QMITKNIFTYVIEWAPPLICATIONPLUGIN_H_ */
