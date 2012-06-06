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
  this->m_Inverse = false;
  this->SetMinimum(0);
  this->SetMaximum(100);
  this->SetValue(0);
  this->SetText("Value");

  this->gridLayout->setColumnStretch(0, 0);
  this->gridLayout->setColumnStretch(1, 1000);
  this->gridLayout->setColumnStretch(0, 0);

  connect(this->spinBox, SIGNAL(valueChanged(int)), this, SLOT(SetValueOnSlider(int)));
  connect(this->horizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(SetValueOnSpinBox(int)));
}

IntegerSpinBoxAndSliderWidget::~IntegerSpinBoxAndSliderWidget()
{

}

void IntegerSpinBoxAndSliderWidget::SetOffset(int i)
{
  this->m_Offset = i;
}

int IntegerSpinBoxAndSliderWidget::GetOffset() const
{
  return m_Offset;
}

void IntegerSpinBoxAndSliderWidget::SetInverse(bool b)
{
  this->m_Inverse = b;
}

bool IntegerSpinBoxAndSliderWidget::GetInverse() const
{
  return this->m_Inverse;
}

int IntegerSpinBoxAndSliderWidget::GetMinimumWithoutOffset() const
{
  return this->spinBox->minimum() - m_Offset;
}

void IntegerSpinBoxAndSliderWidget::SetMinimumWithoutOffset(int i)
{
  int minimumOnWidgets = i + m_Offset;
  this->m_PreviousMinimum = this->spinBox->minimum() - m_Offset;
  this->SetBlockSignals(true);
  if (this->spinBox->minimum() != minimumOnWidgets)
  {
    this->spinBox->setMinimum(minimumOnWidgets);
  }
  if (this->horizontalSlider->minimum() != minimumOnWidgets)
  {
    this->horizontalSlider->setMinimum(minimumOnWidgets);
  }
  this->SetBlockSignals(false);
}

int IntegerSpinBoxAndSliderWidget::GetMaximumWithoutOffset() const
{
  return this->spinBox->maximum() - m_Offset;
}

void IntegerSpinBoxAndSliderWidget::SetMaximumWithoutOffset(int i)
{
  int maximumOnWidgets = i + m_Offset;
  this->m_PreviousMaximum = this->spinBox->maximum() - m_Offset;
  this->SetBlockSignals(true);
  if (this->spinBox->maximum() != maximumOnWidgets)
  {
    this->spinBox->setMaximum(maximumOnWidgets);
  }
  if (this->horizontalSlider->maximum() != maximumOnWidgets)
  {
    this->horizontalSlider->setMaximum(maximumOnWidgets);
  }
  this->SetBlockSignals(false);
}

int IntegerSpinBoxAndSliderWidget::GetValueWithoutOffset() const
{
  int value = this->spinBox->value() - m_Offset;

  if (m_Inverse)
  {
    int minimum = this->GetMinimumWithoutOffset();
    int maximum = this->GetMaximumWithoutOffset();

    value = maximum - (value - minimum);
  }

  return value;
}

void IntegerSpinBoxAndSliderWidget::SetValueWithoutOffset(int i)
{
  int valueOnWidgets = i + m_Offset;
  this->m_PreviousValue = this->spinBox->value() - m_Offset;

  if (m_Inverse)
  {
    int minimum = this->GetMinimumWithoutOffset();
    int maximum = this->GetMaximumWithoutOffset();

    valueOnWidgets = (maximum - (i - minimum)) + m_Offset;
    this->m_PreviousValue =  maximum - (this->m_PreviousValue - minimum);
  }

  this->SetBlockSignals(true);
  if (this->spinBox->value() != valueOnWidgets)
  {
    this->spinBox->setValue(valueOnWidgets);
  }
  if (this->horizontalSlider->value() != valueOnWidgets)
  {
    this->horizontalSlider->setValue(valueOnWidgets);
  }
  this->SetBlockSignals(false);
}

void IntegerSpinBoxAndSliderWidget::EmitCurrentValues()
{
  int value = this->GetValueWithoutOffset();

  if (m_Inverse)
  {
    int minimum = this->GetMinimumWithoutOffset();
    int maximum = this->GetMaximumWithoutOffset();

    emit IntegerValueChanged(maximum - (this->m_PreviousValue - minimum), maximum - (value - minimum));
  }
  else
  {
    emit IntegerValueChanged(this->m_PreviousValue, value);
  }
}

void IntegerSpinBoxAndSliderWidget::SetValueOnSpinBox(int i)
{
  // Called from the slider, so we set the spin box to match, and input will include offset.
  int valueWithoutOffset = i - m_Offset;
  this->SetValueWithoutOffset(valueWithoutOffset);
  this->EmitCurrentValues();
}

void IntegerSpinBoxAndSliderWidget::SetValueOnSlider(int i)
{
  // Called from the spin box, so we set the slider to match, and input will include offset.
  int valueWithoutOffset = i - m_Offset;
  this->SetValueWithoutOffset(valueWithoutOffset);
  this->EmitCurrentValues();
}

void IntegerSpinBoxAndSliderWidget::SetMinimum(int min)
{
  this->SetMinimumWithoutOffset(min);
}

int IntegerSpinBoxAndSliderWidget::GetMinimum() const
{
  return this->GetMinimumWithoutOffset();
}

void IntegerSpinBoxAndSliderWidget::SetMaximum(int max)
{
  this->SetMaximumWithoutOffset(max);
}

int IntegerSpinBoxAndSliderWidget::GetMaximum() const
{
  return this->GetMaximumWithoutOffset();
}

void IntegerSpinBoxAndSliderWidget::SetValue(int value)
{
  int tmp = this->ClampValueToWithinRange(value);
  this->SetValueWithoutOffset(tmp);
}

int IntegerSpinBoxAndSliderWidget::GetValue() const
{
  return this->GetValueWithoutOffset();
}

int IntegerSpinBoxAndSliderWidget::ClampValueToWithinRange(int i)
{
  int tmp = i;
  if (tmp < this->GetMinimumWithoutOffset())
  {
    tmp = this->GetMinimumWithoutOffset();
  }
  else if (tmp > this->GetMaximumWithoutOffset())
  {
    tmp = this->GetMaximumWithoutOffset();
  }
  return tmp;
}

void IntegerSpinBoxAndSliderWidget::SetText(QString text)
{
  this->label->setText(text);
}

void IntegerSpinBoxAndSliderWidget::SetContentsMargins(int margin)
{
  gridLayout->setContentsMargins(margin, margin, margin, margin);
}

void IntegerSpinBoxAndSliderWidget::SetSpacing(int spacing)
{
  gridLayout->setSpacing(spacing);
}

void IntegerSpinBoxAndSliderWidget::SetBlockSignals(bool b)
{
  this->spinBox->blockSignals(b);
  this->horizontalSlider->blockSignals(b);
  this->label->blockSignals(b);
}

void IntegerSpinBoxAndSliderWidget::SetEnabled(bool b)
{
  this->spinBox->setEnabled(b);
  this->horizontalSlider->setEnabled(b);
  this->label->setEnabled(b);
}

bool IntegerSpinBoxAndSliderWidget::GetEnabled() const
{
  return this->horizontalSlider->isEnabled();
}

#endif



