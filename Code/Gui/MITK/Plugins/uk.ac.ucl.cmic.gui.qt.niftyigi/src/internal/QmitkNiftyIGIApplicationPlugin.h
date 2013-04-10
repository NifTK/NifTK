/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKNIFTYIGIAPPLICATIONPLUGIN_H_
#define QMITKNIFTYIGIAPPLICATIONPLUGIN_H_

#include <berryAbstractUICTKPlugin.h>
#include "QmitkCommonAppsApplicationPlugin.h"

/**
 * \class QmitkNiftyIGIApplicationPlugin
 * \brief Implements QT and CTK specific functionality to launch the application as a plugin.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyigi_internal
 */
class QmitkNiftyIGIApplicationPlugin : public QmitkCommonAppsApplicationPlugin, public berry::AbstractUICTKPlugin
{
  Q_OBJECT
  
public:

  QmitkNiftyIGIApplicationPlugin();
  ~QmitkNiftyIGIApplicationPlugin();

  void start(ctkPluginContext*);
  void stop(ctkPluginContext*);

protected:

  /// \brief Called by framework to get a URL for help system.
  virtual QString GetHelpHomePageURL() const;

private:

};

#endif /* QMITKNIFTYIGIAPPLICATIONPLUGIN_H_ */
