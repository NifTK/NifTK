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
#ifndef SLICESELECTWIDGET_CPP
#define SLICESELECTWIDGET_CPP

#include "SliceSelectWidget.h"

const QString SliceSelectWidget::OBJECT_NAME = QString("SliceSelectWidget");

SliceSelectWidget::SliceSelectWidget(QWidget *parent)
: IntegerSpinBoxAndSliderWidget(parent)
{
  this->setObjectName(OBJECT_NAME);
  this->setWindowTitle(tr("Slice Selection Controller"));

  this->m_Offset = 1;
  this->SetMinimum(0);
  this->SetMaximum(100);
  this->SetValue(0);
  this->SetText(tr("slice"));

  connect(this->spinBox, SIGNAL(valueChanged(int)), this, SLOT(OnChangeSliceNumber(int)));
  connect(this->horizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(OnChangeSliceNumber(int)));
}

SliceSelectWidget::~SliceSelectWidget()
{

}

void SliceSelectWidget::SetSliceNumber(int value)
{
  this->SetValue(value);
}

void SliceSelectWidget::OnChangeSliceNumber(int value)
{
  emit SliceNumberChanged(m_PreviousValue - m_Offset, value - m_Offset);
}

void SliceSelectWidget::AddToSliceNumber(int i)
{
  int currentValue = this->GetValue();
  int nextValue = currentValue + i;

  if (nextValue >= this->GetMinimum() && nextValue <= this->GetMaximum())
  {
    this->horizontalSlider->setValue(nextValue + m_Offset);
  }
}

#endif
