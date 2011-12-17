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
#ifndef DOUBLESPINBOXANDSLIDERWIDGET_CPP
#define DOUBLESPINBOXANDSLIDERWIDGET_CPP

#include "DoubleSpinBoxAndSliderWidget.h"
#include "WidgetHelper.h"
#include <iostream>
DoubleSpinBoxAndSliderWidget::DoubleSpinBoxAndSliderWidget(QWidget *parent)
{
  this->setupUi(this);
  this->SetText("Value");

  this->m_SliderMin = 0;
  this->m_SliderMax = 1000;

  this->horizontalSlider->setMinimum(m_SliderMin);
  this->horizontalSlider->setMaximum(m_SliderMax);

  this->SetMinimum(0);
  this->SetMaximum(100);
  this->SetValue(0);

  this->m_PreviousMinimum =  this->GetMinimum();
  this->m_PreviousMaximum =  this->GetMaximum();
  this->m_PreviousValue = this->GetValue();

  connect(this->spinBox, SIGNAL(valueChanged(double)), this, SLOT(SetValueOnSlider(double)));
  connect(this->horizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(SetValueOnSpinBox(int)));
}

DoubleSpinBoxAndSliderWidget::~DoubleSpinBoxAndSliderWidget()
{

}

void DoubleSpinBoxAndSliderWidget::SetMinimum(double min)
{
  this->m_PreviousMinimum = this->spinBox->minimum();
  this->spinBox->blockSignals(true);
  this->spinBox->setMinimum(min);
  this->spinBox->blockSignals(false);
}

void DoubleSpinBoxAndSliderWidget::SetMaximum(double max)
{
  this->m_PreviousMaximum = this->spinBox->maximum();
  this->spinBox->blockSignals(true);
  this->spinBox->setMaximum(max);
  this->spinBox->blockSignals(false);
}

void DoubleSpinBoxAndSliderWidget::SetText(QString text)
{
  this->label->setText(text);
}

double DoubleSpinBoxAndSliderWidget::ClampValueToWithinRange(double i)
{
  double tmp = i;
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

void DoubleSpinBoxAndSliderWidget::SetValue(double value)
{
  this->m_PreviousValue = this->spinBox->value();

  double tmp = this->ClampValueToWithinRange(value);
  int sliderValue = ConvertSpinBoxValueToSliderValue(tmp, this->spinBox->minimum(), this->spinBox->maximum(), m_SliderMin, m_SliderMax);

  this->spinBox->blockSignals(true);
  this->horizontalSlider->blockSignals(true);

  this->spinBox->setValue(tmp);
  this->horizontalSlider->setValue(sliderValue);

  this->spinBox->blockSignals(false);
  this->horizontalSlider->blockSignals(false);

}

double DoubleSpinBoxAndSliderWidget::GetValue() const
{
  return this->spinBox->value();
}

double DoubleSpinBoxAndSliderWidget::GetMinimum() const
{
  return this->spinBox->minimum();
}

double DoubleSpinBoxAndSliderWidget::GetMaximum() const
{
  return this->spinBox->maximum();
}

void DoubleSpinBoxAndSliderWidget::SetValueOnSpinBox(int i)
{
  double spinBoxValue = ConvertSliderValueToSpinBoxValue(i, this->spinBox->minimum(), this->spinBox->maximum(), m_SliderMin, m_SliderMax);
  this->m_PreviousValue = this->spinBox->value();
  this->spinBox->blockSignals(true);
  this->spinBox->setValue(spinBoxValue);
  this->spinBox->blockSignals(false);
  emit DoubleValueChanged(this->m_PreviousValue, spinBoxValue);
}

void DoubleSpinBoxAndSliderWidget::SetValueOnSlider(double i)
{
  this->m_PreviousValue = ConvertSpinBoxValueToSliderValue(this->horizontalSlider->value(), this->spinBox->minimum(), this->spinBox->maximum(), m_SliderMin, m_SliderMax);
  int sliderValue = ConvertSpinBoxValueToSliderValue(i, this->spinBox->minimum(), this->spinBox->maximum(), m_SliderMin, m_SliderMax);
  this->horizontalSlider->blockSignals(true);
  this->horizontalSlider->setValue(sliderValue);
  this->horizontalSlider->blockSignals(false);
  emit DoubleValueChanged(this->m_PreviousValue, i);
}

#endif



