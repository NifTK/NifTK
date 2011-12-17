/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-18 12:23:46 +0100 (Thu, 18 Aug 2011) $
 Revision          : $Revision: 7128 $
 Last modified by  : $Author: ad $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef HELPABOUTDIALOG_H
#define HELPABOUTDIALOG_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_HelpAboutDialog.h"
#include <QDialog>

/**
 * \class HelpAboutDialog
 */
class NIFTKQT_WINEXPORT HelpAboutDialog : public QDialog, public Ui_HelpAboutDialog {

	Q_OBJECT

public:

	/** Default constructor. */
	HelpAboutDialog(QWidget *parent, QString applicationName);

	/** Destructor. */
	~HelpAboutDialog();

private:

	HelpAboutDialog(const HelpAboutDialog&);  // Purposefully not implemented.
	void operator=(const HelpAboutDialog&);  // Purposefully not implemented.

  void SetHelpAboutText(QString applicationName);

  QString m_ApplicationName;

};
#endif

