/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 15:53:45 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7857 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QMITKHELPABOUTDIALOG_H
#define QMITKHELPABOUTDIALOG_H

#include "niftkQmitkExtExports.h"

#include "ui_QmitkHelpAboutDialog.h"
#include <QDialog>

/**
 * \class HelpAboutDialog
 * \brief Prints out a useful About dialog with software version numbers.
 */
class NIFTKQMITKEXT_EXPORT QmitkHelpAboutDialog : public QDialog, public Ui_QmitkHelpAboutDialog {

	Q_OBJECT

public:

	/// \brief Constructor with additional name parameter.
	QmitkHelpAboutDialog(QWidget *parent, QString applicationName);

	/// \brief destructor.
	~QmitkHelpAboutDialog();

private:

	QmitkHelpAboutDialog(const QmitkHelpAboutDialog&);  // Purposefully not implemented.
	void operator=(const QmitkHelpAboutDialog&);  // Purposefully not implemented.

	// Generates the text, taking the application name, and internally lots of values configured from NifTKConfigure.h
  void GenerateHelpAboutText(QString applicationName);

  QString m_ApplicationName;
};
#endif

