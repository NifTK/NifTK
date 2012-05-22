/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-01 16:14:30 +0100 (Mon, 01 Aug 2011) $
 Revision          : $Revision: 6909 $
 Last modified by  : $Author: ad $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "QmitkNiftyRegView.h"

// Qt
#include <QMessageBox>

const std::string QmitkNiftyRegView::VIEW_ID = "uk.ac.ucl.cmic.views.niftyregview";

QmitkNiftyRegView::QmitkNiftyRegView()
{
}

QmitkNiftyRegView::~QmitkNiftyRegView()
{
}

void QmitkNiftyRegView::CreateQtPartControl(QWidget *parent)
{
  // create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi( parent );
}

void QmitkNiftyRegView::SetFocus()
{
}
