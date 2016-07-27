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

#include <niftkBaseApplicationPluginActivator.h>

/**
 * \class QmitkNiftyMIDASApplicationPlugin
 * \brief Implements QT and CTK specific functionality to launch the application as a plugin.
 * \ingroup uk_ac_ucl_cmic_niftymidas_internal
 */
class QmitkNiftyMIDASApplicationPlugin : public BaseApplicationPluginActivator
{
  Q_OBJECT
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
  Q_PLUGIN_METADATA(IID "uk_ac_ucl_cmic_niftymidas")
#endif
  
public:

  QmitkNiftyMIDASApplicationPlugin();
  ~QmitkNiftyMIDASApplicationPlugin();

  virtual void start(ctkPluginContext*) override;
  virtual void stop(ctkPluginContext*) override;

protected:

  /// \brief Called by framework to get a URL for help system.
  virtual QString GetHelpHomePageURL() const;

private:

};

#endif
