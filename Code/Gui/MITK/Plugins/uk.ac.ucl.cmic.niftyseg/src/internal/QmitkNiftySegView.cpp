/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-01 17:38:45 +0100 (Mon, 01 Aug 2011) $
 Revision          : $Revision: 6915 $
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
#include "QmitkNiftySegView.h"
#include "QmitkStdMultiWidget.h"

// Qt
#include <QMessageBox>

const std::string QmitkNiftySegView::VIEW_ID = "uk.ac.ucl.cmic.views.niftysegview";

QmitkNiftySegView::QmitkNiftySegView()
: QmitkFunctionality()
{
}

QmitkNiftySegView::~QmitkNiftySegView()
{
}

void QmitkNiftySegView::CreateQtPartControl(QWidget *parent)
{
  // create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi( parent );

  //Default settings
  m_Controls.m_EMPriorsImagesComboBox->setDisabled(true);
  m_Controls.m_EMPriorsImagesBrowsePushButton->setDisabled(true);
  m_Controls.m_EMMeansTextEdit->setDisabled(true);

  connect(m_Controls.m_EMAutomaticRadioButton, SIGNAL(clicked(bool)), this, SLOT(OnClickedEMInitialisationRadioButtons(bool)));
  connect(m_Controls.m_EMPriorsRadioButton, SIGNAL(clicked(bool)), this, SLOT(OnClickedEMInitialisationRadioButtons(bool)));
  connect(m_Controls.m_EMMeansRadioButton, SIGNAL(clicked(bool)), this, SLOT(OnClickedEMInitialisationRadioButtons(bool)));
}

void QmitkNiftySegView::OnClickedEMInitialisationRadioButtons(bool bClicked)
{
  if(m_Controls.m_EMAutomaticRadioButton->isChecked())
  {
    m_Controls.m_EMPriorsImagesComboBox->setDisabled(true);
    m_Controls.m_EMPriorsImagesBrowsePushButton->setDisabled(true);
    m_Controls.m_EMMeansTextEdit->setDisabled(true);
  }
  else if(m_Controls.m_EMPriorsRadioButton->isChecked())
  {
    m_Controls.m_EMPriorsImagesComboBox->setEnabled(true);
    m_Controls.m_EMPriorsImagesBrowsePushButton->setEnabled(true);
    m_Controls.m_EMMeansTextEdit->setEnabled(false);
  }
  else if(m_Controls.m_EMMeansRadioButton->isChecked())
  {
    m_Controls.m_EMPriorsImagesComboBox->setDisabled(true);
    m_Controls.m_EMPriorsImagesBrowsePushButton->setDisabled(true);
    m_Controls.m_EMMeansTextEdit->setEnabled(true);
  }

}

