/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkNiftyMIDASApplicationPlugin_h
#define QmitkNiftyMIDASApplicationPlugin_h

#include <berryAbstractUICTKPlugin.h>
#include <QmitkCommonAppsApplicationPlugin.h>

/**
 * \class QmitkNiftyMIDASApplicationPlugin
 * \brief Implements QT and CTK specific functionality to launch the application as a plugin.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftymidas_internal
 */
class QmitkNiftyMIDASApplicationPlugin : public QmitkCommonAppsApplicationPlugin, public berry::AbstractUICTKPlugin
{
  Q_OBJECT
  
public:

  QmitkNiftyMIDASApplicationPlugin();
  ~QmitkNiftyMIDASApplicationPlugin();

  void start(ctkPluginContext*);
  void stop(ctkPluginContext*);

protected:

  /// \brief Called each time a data node is added, so we make sure it is initialised with a Window/Level.
  virtual void NodeAdded(const mitk::DataNode *node);

  /// \brief Called by framework to get a URL for help system.
  virtual QString GetHelpHomePageURL() const;

private:

};

#endif /* QMITKNIFTYMIDASAPPLICATIONPLUGIN_H_ */
