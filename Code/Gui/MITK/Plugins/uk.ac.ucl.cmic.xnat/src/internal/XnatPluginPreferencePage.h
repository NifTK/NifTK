/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatPluginPreferencePage_h
#define XnatPluginPreferencePage_h

#include "ui_XnatPluginPreferencePage.h"

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QRadioButton;
class QDoubleSpinBox;
class QSpinBox;

/**
 * \class XnatPluginPreferencePage
 * \brief Preferences page for this plugin.
 * \ingroup uk_ac_ucl_cmic_gui_xnat
 *
 */
class XnatPluginPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:
  static const std::string SERVER_NAME;
  static const std::string SERVER_DEFAULT;
  static const std::string USER_NAME;
  static const std::string USER_DEFAULT;
  static const std::string DOWNLOAD_DIRECTORY_NAME;
  static const std::string DOWNLOAD_DIRECTORY_DEFAULT;
  static const std::string WORK_DIRECTORY_NAME;
  static const std::string WORK_DIRECTORY_DEFAULT;

  explicit XnatPluginPreferencePage();
  virtual ~XnatPluginPreferencePage();

  void Init(berry::IWorkbench::Pointer workbench);

  void CreateQtControl(QWidget* widget);

  QWidget* GetQtControl() const;

  ///
  /// \see IPreferencePage::PerformOk()
  ///
  virtual bool PerformOk();

  ///
  /// \see IPreferencePage::PerformCancel()
  ///
  virtual void PerformCancel();

  ///
  /// \see IPreferencePage::Update()
  ///
  virtual void Update();

protected slots:

private:

  bool m_Initializing;

  berry::IPreferences::Pointer m_XnatBrowserViewPreferences;

  QWidget* m_MainControl;

  /// \brief All the controls for the main view part.
  Ui::XnatPluginPreferencePage* m_Controls;

  Q_DISABLE_COPY(XnatPluginPreferencePage);
};

#endif
