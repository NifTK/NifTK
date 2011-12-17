/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-02 17:25:37 +0100 (Thu, 02 Sep 2010) $
 Revision          : $Revision: 6276 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef TOGGLEWIDGET_CPP
#define TOGGLEWIDGET_CPP

#include "ToggleWidget.h"

const QString ToggleWidget::OBJECT_NAME = QString("ToggleWidget");

ToggleWidget::ToggleWidget(QWidget *parent)
{
  this->setupUi(this);
  this->setWindowTitle(tr("Toggle Controller"));
  this->setObjectName(OBJECT_NAME);
  this->m_PreviousIndex = 0;
  this->m_PreviousText = "";

  connect(this->togglePushButton, SIGNAL(pressed()), this, SLOT(OnButtonPressed()));
}

ToggleWidget::~ToggleWidget()
{

}

void ToggleWidget::ClearList()
{
  this->fileComboBox->clear();
}

void ToggleWidget::SetDataSets(QStringList list)
{
  this->ClearList();

  for (int i = 0; i < list.size(); i++)
  {
    this->fileComboBox->addItem(list[i]);
  }
}

void ToggleWidget::SetCurrentIndex(int i)
{
  this->fileComboBox->setCurrentIndex(i);
}

void ToggleWidget::SetCurrentIndex(QString name)
{
  for (int i = 0; i < this->fileComboBox->count(); i++)
  {
    if (this->fileComboBox->itemText(i) == name)
    {
      this->fileComboBox->setCurrentIndex(i);
      break;
    }
  }
}

void ToggleWidget::OnButtonPressed()
{
  this->m_PreviousIndex = this->fileComboBox->currentIndex();
  this->m_PreviousText = this->fileComboBox->currentText();

  int nextIndex = this->m_PreviousIndex + 1;
  if (nextIndex >= this->fileComboBox->count())
  {
    nextIndex = 0;
  }

  this->fileComboBox->setCurrentIndex(nextIndex);

  emit DataSetToggled(this->m_PreviousIndex, nextIndex, this->m_PreviousText, this->fileComboBox->currentText());
}

void ToggleWidget::SetDataSet(int position, QString text)
{
  if (position >= this->fileComboBox->count())
  {
    for (int i = this->fileComboBox->count(); i < position; i++)
    {
      this->fileComboBox->insertItem(position, "");
    }
    this->fileComboBox->insertItem(position, text);
  }
  else
  {
    this->fileComboBox->setItemText(position, text);
  }

  this->fileComboBox->setCurrentIndex(position);
}
#endif
