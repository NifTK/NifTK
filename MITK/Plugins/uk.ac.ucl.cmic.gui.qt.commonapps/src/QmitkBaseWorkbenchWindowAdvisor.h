/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkBaseWorkbenchWindowAdvisor_h
#define QmitkBaseWorkbenchWindowAdvisor_h

#include <uk_ac_ucl_cmic_gui_qt_commonapps_Export.h>
#include <QmitkExtWorkbenchWindowAdvisor.h>

/**
 * \class QmitkBaseViewWorkbenchWindowAdvisor
 * \brief Abstract advisor class to set up application windows on startup.
 * \ingroup uk_ac_ucl_cmic_gui_qt_commonapps
 * \sa QmitkHelpAboutDialog
 */
class CMIC_QT_COMMONAPPS QmitkBaseWorkbenchWindowAdvisor : public QmitkExtWorkbenchWindowAdvisor
{
  Q_OBJECT

public:

  QmitkBaseWorkbenchWindowAdvisor(berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer);

  /// \brief We override the base class PreWindowOpen to specifically set
  /// QmitkExtWorkbenchWindowAdvisor::showVersionInfo and
  /// QmitkExtWorkbenchWindowAdvisor::showMitkVersionInfo to false.
  virtual void PreWindowOpen();

  /// \brief We override the base class PostWindowCreate to customise
  /// the About dialog box to call our QmitkHelpAboutDialog, and to remove
  /// the Welcome dialog box.
  virtual void PostWindowCreate();

public slots:

  /// \brief Opens the Help About dialog box.
  void OnHelpAbout();

protected:

  /**
   * \brief Directly tries to open the given editor: use sparingly.
   */
  void OpenEditor(const QString& editorName);

  /**
   * \brief Checks if the environment variable contains a string value "ON" or "1", and if so tries to open the given editor.
   */
  void OpenEditorIfEnvironmentVariableIsON(const std::string& envVariable, const QString& editorName);
};

#endif /*QMITKBASEWORKBENCHWINDOWADVISOR_H_*/

