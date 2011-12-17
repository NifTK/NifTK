/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 7658 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef INTEGERSPINBOXANDSLIDERWIDGET_CPP
#define INTEGERSPINBOXANDSLIDERWIDGET_CPP

#include "IntegerSpinBoxAndSliderWidget.h"

IntegerSpinBoxAndSliderWidget::IntegerSpinBoxAndSliderWidget(QWidget *parent)
{
  this->setupUi(this);
  this->m_Offset = 0;
  this->SetMinimum(0);
  this->SetMaximum(100);
  this->SetValue(0);
  this->SetText("Value");

  connect(this->spinBox, SIGNAL(valueChanged(int)), this, SLOT(SetValueOnSlider(int)));
  connect(this->horizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(SetValueOnSpinBox(int)));
}

IntegerSpinBoxAndSliderWidget::~IntegerSpinBoxAndSliderWidget()
{

}

void IntegerSpinBoxAndSliderWidget::SetMinimum(int min)
{
  this->m_PreviousMinimum = this->spinBox->minimum();
  this->spinBox->setMinimum(min + m_Offset);
  this->horizontalSlider->setMinimum(min + m_Offset);
}

void IntegerSpinBoxAndSliderWidget::SetMaximum(int max)
{
  this->m_PreviousMaximum = this->spinBox->maximum();
  this->spinBox->setMaximum(max + m_Offset);
  this->horizontalSlider->setMaximum(max + m_Offset);
}

void IntegerSpinBoxAndSliderWidget::SetText(QString text)
{
  this->label->setText(text);
}

int IntegerSpinBoxAndSliderWidget::ClampValueToWithinRange(int i)
{
  int tmp = i;
  if (tmp < this->spinBox->minimum())
  {
    tmp = this->spinBox->minimum();
  }
  else if (tmp > this->spinBox->maximum())
  {
    tmp = this->spinBox->maximum();
  }
  return tmp;
}

void IntegerSpinBoxAndSliderWidget::SetValue(int value)
{
  this->m_PreviousValue = this->spinBox->value();

  int tmp = this->ClampValueToWithinRange(value + m_Offset);
  this->spinBox->blockSignals(true);
  this->horizontalSlider->blockSignals(true);
  this->spinBox->setValue(tmp);
  this->horizontalSlider->setValue(tmp);
  this->spinBox->blockSignals(false);
  this->horizontalSlider->blockSignals(false);

}

int IntegerSpinBoxAndSliderWidget::GetValue() const
{
  return this->spinBox->value() - m_Offset;
}

int IntegerSpinBoxAndSliderWidget::GetMinimum() const
{
  return this->spinBox->minimum() - m_Offset;
}

int IntegerSpinBoxAndSliderWidget::GetMaximum() const
{
  return this->spinBox->maximum() - m_Offset;
}

void IntegerSpinBoxAndSliderWidget::SetValueOnSpinBox(int i)
{
  //set the value on the spin box
  this->m_PreviousValue = this->spinBox->value();
  this->spinBox->blockSignals(true);
  this->spinBox->setValue(i);
  this->spinBox->blockSignals(false);
  emit IntegerValueChanged(this->m_PreviousValue, i);
}

void IntegerSpinBoxAndSliderWidget::SetValueOnSlider(int i)
{
  //set the value on the slider
  this->m_PreviousValue = this->horizontalSlider->value();
  this->horizontalSlider->blockSignals(true);
  this->horizontalSlider->setValue(i);
  this->horizontalSlider->blockSignals(false);
  emit IntegerValueChanged(this->m_PreviousValue, i);
}

void IntegerSpinBoxAndSliderWidget::SetOffset(int i)
{
  this->m_Offset = i;
}

int IntegerSpinBoxAndSliderWidget::GetOffset() const
{
  return m_Offset;
}

void IntegerSpinBoxAndSliderWidget::SetContentsMargins(int margin)
{
  gridLayout->setContentsMargins(margin, margin, margin, margin);
}

void IntegerSpinBoxAndSliderWidget::SetSpacing(int spacing)
{
  gridLayout->setSpacing(spacing);
}

#endif



