/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "QmitkNiftySegView.h"

// Qt
#include <QMessageBox>

const std::string QmitkNiftySegView::VIEW_ID = "uk.ac.ucl.cmic.views.niftysegview";

QmitkNiftySegView::QmitkNiftySegView()
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

void QmitkNiftySegView::SetFocus()
{
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

