/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkNiftyViewApplicationPlugin_h
#define QmitkNiftyViewApplicationPlugin_h

#include <QmitkCommonAppsApplicationPlugin.h>

/**
 * \class QmitkNiftyViewApplicationPlugin
 * \brief Implements QT and CTK specific functionality to launch the application as a plugin.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview_internal
 */
class QmitkNiftyViewApplicationPlugin : public QmitkCommonAppsApplicationPlugin
{
  Q_OBJECT
  
public:

  QmitkNiftyViewApplicationPlugin();
  ~QmitkNiftyViewApplicationPlugin();

  virtual void start(ctkPluginContext*);
  virtual void stop(ctkPluginContext*);

protected:

  /// \brief Called by framework to get a URL for help system.
  virtual QString GetHelpHomePageURL() const;

private:

};

#endif /* QMITKNIFTYVIEWAPPLICATIONPLUGIN_H_ */
