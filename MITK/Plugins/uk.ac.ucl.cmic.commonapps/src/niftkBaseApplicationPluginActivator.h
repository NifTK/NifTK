/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBaseApplicationPluginActivator_h
#define niftkBaseApplicationPluginActivator_h

#include <uk_ac_ucl_cmic_commonapps_Export.h>

#include <ctkServiceTracker.h>

#include <QObject>
#include <ctkPluginActivator.h>


namespace niftk
{

/// \class BaseApplicationPluginActivator
/// \brief Abstract class that implements QT and CTK specific functionality to launch the application as a plugin.
/// \ingroup uk_ac_ucl_cmic_commonapps
class COMMONAPPS_EXPORT BaseApplicationPluginActivator : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  BaseApplicationPluginActivator();
  virtual ~BaseApplicationPluginActivator();

  static BaseApplicationPluginActivator* GetInstance();

  ctkPluginContext* GetContext() const;

  virtual void start(ctkPluginContext* context) override;

  virtual void stop(ctkPluginContext* context) override;

protected:

  /// \brief Deliberately not virtual method thats called by derived classes within the start method to set up the help system.
  void RegisterHelpSystem();

  // \brief Sets a preference whether to reinitialise the rendering manager after opening a file.
  // It is suggested to set this to 'false' with the DnD display.
  void SetFileOpenTriggersReinit(bool openEditor);

  /// \brief Derived classes should provide a URL for which help page to use as the 'home' page.
  virtual QString GetHelpHomePageURL() const
  {
    return QString();
  }

private:

  ctkPluginContext* m_Context;

  static BaseApplicationPluginActivator* s_Instance;

};

}

#endif
