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
#ifndef ROWSANDCOLUMNSWIDGET_CPP
#define ROWSANDCOLUMNSWIDGET_CPP

#include "RowsAndColumnsWidget.h"

const QString RowsAndColumnsWidget::OBJECT_NAME = QString("RowsAndColumnsWidget");

RowsAndColumnsWidget::RowsAndColumnsWidget(QWidget *parent)
{
  this->setupUi(this);
  this->setObjectName(OBJECT_NAME);
  this->rowsSpinBox->setValue(1);
  this->columnsSpinBox->setValue(1);
  this->m_PreviousNumberOfColumns = 1;
  this->m_PreviousNumberOfRows = 1;
  this->SetMinimumNumberOfRows(1);
  this->SetMinimumNumberOfColumns(1);
  this->SetMaximumNumberOfRows(5);
  this->SetMaximumNumberOfColumns(5);

  connect(this->rowsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnNumberOfRowsChanged(int)));
  connect(this->columnsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnNumberOfColumnsChanged(int)));

}

RowsAndColumnsWidget::~RowsAndColumnsWidget()
{
}

void RowsAndColumnsWidget::SetNumberOfRows(int i)
{
  this->rowsSpinBox->blockSignals(true);
  this->rowsSpinBox->setValue(i);
  this->rowsSpinBox->blockSignals(false);
  this->m_PreviousNumberOfRows = this->GetNumberOfRows();
}

int RowsAndColumnsWidget::GetNumberOfRows() const
{
  return this->rowsSpinBox->value();
}

void RowsAndColumnsWidget::SetNumberOfColumns(int i)
{
  this->columnsSpinBox->blockSignals(true);
  this->columnsSpinBox->setValue(i);
  this->columnsSpinBox->blockSignals(false);
  this->m_PreviousNumberOfColumns = this->GetNumberOfColumns();
}

int RowsAndColumnsWidget::GetNumberOfColumns() const
{
  return this->columnsSpinBox->value();
}

void RowsAndColumnsWidget::OnNumberOfRowsChanged(int i)
{
  emit NumberOfRowsChanged(this->m_PreviousNumberOfRows, i);
  this->m_PreviousNumberOfRows = i;
}

void RowsAndColumnsWidget::OnNumberOfColumnsChanged(int i)
{
  emit NumberOfColumnsChanged(this->m_PreviousNumberOfColumns, i);
  this->m_PreviousNumberOfColumns = i;
}

void RowsAndColumnsWidget::SetMinimumNumberOfRows(int i)
{
  this->rowsSpinBox->setMinimum(i);
}

int RowsAndColumnsWidget::GetMinimumNumberOfRows() const
{
  return this->rowsSpinBox->minimum();
}

void RowsAndColumnsWidget::SetMinimumNumberOfColumns(int i)
{
  this->columnsSpinBox->setMinimum(i);
}

int RowsAndColumnsWidget::GetMinimumNumberOfColumns() const
{
  return this->columnsSpinBox->minimum();
}

void RowsAndColumnsWidget::SetMaximumNumberOfRows(int i)
{
  this->rowsSpinBox->setMaximum(i);
}

int RowsAndColumnsWidget::GetMaximumNumberOfRows() const
{
  return this->rowsSpinBox->maximum();
}

void RowsAndColumnsWidget::SetMaximumNumberOfColumns(int i)
{
  this->columnsSpinBox->setMaximum(i);
}

int RowsAndColumnsWidget::GetMaximumNumberOfColumns() const
{
  return this->columnsSpinBox->maximum();
}

#endif
