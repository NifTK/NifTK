/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-26 14:06:43 +0100 (Thu, 26 May 2011) $
 Revision          : $Revision: 6276 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef LISTVIEWOKCANCELDIALOG_CPP
#define LISTVIEWOKCANCELDIALOG_CPP

#include "ListViewOKCancelDialog.h"

ListViewOKCancelDialog::ListViewOKCancelDialog(QWidget *parent)
{
	this->setupUi(this);
}

ListViewOKCancelDialog::~ListViewOKCancelDialog()
{

}

void ListViewOKCancelDialog::AddItem(QString item)
{
	listWidget->addItem(item);
}

QString ListViewOKCancelDialog::GetSelectedString() const
{
	int currentRow = this->GetSelectedInt();
	return listWidget->item(currentRow)->text();
}

int ListViewOKCancelDialog::GetSelectedInt() const
{
	return listWidget->currentRow();
}

#endif
