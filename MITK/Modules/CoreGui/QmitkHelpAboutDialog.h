/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkHelpAboutDialog_h
#define QmitkHelpAboutDialog_h

#include "niftkCoreGuiExports.h"
#include "ui_QmitkHelpAboutDialog.h"
#include <QDialog>

/**
 * \class HelpAboutDialog
 * \brief Prints out a useful About dialog with software version numbers.
 */
class NIFTKCOREGUI_EXPORT QmitkHelpAboutDialog : public QDialog, public Ui_QmitkHelpAboutDialog {

	Q_OBJECT

public:

  /**
   * \brief Constructor with additional name parameter.
   */
	QmitkHelpAboutDialog(QWidget *parent, QString applicationName);

  /**
   * \brief Destructor.
   */
  virtual ~QmitkHelpAboutDialog();

private:

	QmitkHelpAboutDialog(const QmitkHelpAboutDialog&);  // Purposefully not implemented.
	void operator=(const QmitkHelpAboutDialog&);  // Purposefully not implemented.

	// Generates the text, taking the application name, and internally lots of values configured from NifTKConfigure.h
  void GenerateHelpAboutText(QString applicationName);

  QString m_ApplicationName;
};
#endif

