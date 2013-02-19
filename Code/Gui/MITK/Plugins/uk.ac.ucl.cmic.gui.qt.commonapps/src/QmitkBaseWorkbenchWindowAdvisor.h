/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKBASEWORKBENCHWINDOWADVISOR_H_
#define QMITKBASEWORKBENCHWINDOWADVISOR_H_

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

  /// \brief Checks environment variable NIFTK_MITK_DISPLAY to see
  /// if we are forcing the MITK display open.
  void CheckIfLoadingMITKDisplay();
};

#endif /*QMITKBASEWORKBENCHWINDOWADVISOR_H_*/

