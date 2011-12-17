/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-01 19:03:07 +0100 (Fri, 01 Jul 2011) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef LISTVIEWOKCANCELDIALOG_H
#define LISTVIEWOKCANCELDIALOG_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_ListViewOKCancelDialog.h"
#include <QDialog>
#include <QString>

/**
 * \class ListViewOKCancelDialog
 * \brief presents a list of options, and either returns empty string, or the string that was selected.
 */
class NIFTKQT_WINEXPORT ListViewOKCancelDialog : public QDialog, public Ui_ListViewOKCancelDialog {

	Q_OBJECT

public:

	/** Default constructor. */
	ListViewOKCancelDialog(QWidget *parent);

	/** Destructor. */
	~ListViewOKCancelDialog();

	/** Just to add items to list */
	void AddItem(QString item);

	/** This returns the string thats selected. */
	QString GetSelectedString() const;

	/** This returns the list position that was selected. */
	int GetSelectedInt() const;

private:

	ListViewOKCancelDialog(const ListViewOKCancelDialog&);  // Purposefully not implemented.
	void operator=(const ListViewOKCancelDialog&);  // Purposefully not implemented.

};
#endif

