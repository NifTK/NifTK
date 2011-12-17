/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 6276 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef LOOKUPTABLEWIDGET_CPP
#define LOOKUPTABLEWIDGET_CPP

#include "LookupTableWidget.h"
#include <QComboBox>

const QString LookupTableWidget::OBJECT_NAME = QString("LookupTableWidget");

LookupTableWidget::LookupTableWidget(QWidget *parent)
{
  this->setupUi(this);
  this->setObjectName(OBJECT_NAME);
  this->comboBox->setCurrentIndex(0);
  this->m_PreviousIndex = this->GetCurrentIndex();

  connect(this->comboBox, SIGNAL(activated(int)), this, SLOT(StorePreviousIndex()));
  connect(this->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(TriggerLookupTableIndexChanged()));
}

LookupTableWidget::~LookupTableWidget()
{
}

QComboBox* LookupTableWidget::GetComboBox()
{
  return this->comboBox;
}

int LookupTableWidget::GetCurrentIndex() const
{
  return this->comboBox->currentIndex();
}

void LookupTableWidget::SetCurrentIndex(int i)
{
  this->StorePreviousIndex();
  this->comboBox->blockSignals(true);
  this->comboBox->setCurrentIndex(i);
  this->comboBox->blockSignals(false);
}

void LookupTableWidget::StorePreviousIndex()
{
  this->m_PreviousIndex = this->GetCurrentIndex();
}

void LookupTableWidget::TriggerLookupTableIndexChanged()
{
  QString oldName = this->GetText(this->m_PreviousIndex);
  QString newName = this->GetText(this->GetCurrentIndex());

  emit LookupTableIndexChanged(this->m_PreviousIndex, this->GetCurrentIndex(), oldName, newName);
}

QString LookupTableWidget::GetText(int i) const
{
  return this->comboBox->itemText(i);
}

QStringList LookupTableWidget::GetAllEntries() const
{
  QStringList list;
  for (int i = 0; i < this->comboBox->count(); i++)
  {
    list.push_back(this->GetText(i));
  }
  return list;
}

#endif
