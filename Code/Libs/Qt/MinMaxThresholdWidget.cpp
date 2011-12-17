/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 6840 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MINMAXTHRESHOLDWIDGETWIDGET_CPP
#define MINMAXTHRESHOLDWIDGETWIDGET_CPP

#include "MinMaxThresholdWidget.h"
#include "WidgetHelper.h"
#include "math.h"

const QString MinMaxThresholdWidget::OBJECT_NAME = QString("MinMaxThresholdWidget");

MinMaxThresholdWidget::MinMaxThresholdWidget(QWidget *parent)
{
  this->setupUi(this);
  this->setObjectName(OBJECT_NAME);

  m_SliderMin = 0;
  m_SliderMax = 1000;
  m_PreviousMinimumIntensity = 0;
  m_PreviousMinimumThreshold = 0;
  m_PreviousMaximumIntensity = 255;

  this->minSlider->setMinimum(m_SliderMin);
  this->maxSlider->setMinimum(m_SliderMin);
  this->thresholdSlider->setMinimum(m_SliderMin);

  this->minSlider->setMaximum(m_SliderMax);
  this->maxSlider->setMaximum(m_SliderMax);
  this->thresholdSlider->setMaximum(m_SliderMax);

  this->SetLimits(0, 255);
  this->limitsGroupBox->setChecked(false);
  this->SetLimitsVisible(false);

  connect(this->minSpinBox, SIGNAL(valueChanged(double)), this, SLOT(OnMinimumIntensitySpinBoxChanged(double)));
  connect(this->maxSpinBox, SIGNAL(valueChanged(double)), this, SLOT(OnMaximumIntensitySpinBoxChanged(double)));
  connect(this->thresholdSpinBox, SIGNAL(valueChanged(double)), this, SLOT(OnMinimumThresholdSpinBoxChanged(double)));
  connect(this->minSlider, SIGNAL(valueChanged(int)), this, SLOT(OnMinimumIntensitySliderChanged(int)));
  connect(this->maxSlider, SIGNAL(valueChanged(int)), this, SLOT(OnMaximumIntensitySliderChanged(int)));
  connect(this->thresholdSlider, SIGNAL(valueChanged(int)), this, SLOT(OnMinimumThresholdSliderChanged(int)));
  connect(this->morePushButton, SIGNAL(pressed()), this, SLOT(OnMoreButtonPressed()));
  connect(this->minLimitSpinBox, SIGNAL(valueChanged(double)), this, SLOT(OnMinLimitSpinBoxChanged(double)));
  connect(this->maxLimitSpinBox, SIGNAL(valueChanged(double)), this, SLOT(OnMaxLimitSpinBoxChanged(double)));
}

MinMaxThresholdWidget::~MinMaxThresholdWidget()
{
}

void MinMaxThresholdWidget::BlockAllSignals(bool b)
{
  this->minSpinBox->blockSignals(b);
  this->minSlider->blockSignals(b);
  this->maxSpinBox->blockSignals(b);
  this->maxSlider->blockSignals(b);
  this->thresholdSpinBox->blockSignals(b);
  this->thresholdSlider->blockSignals(b);
  this->minLimitSpinBox->blockSignals(b);
  this->maxLimitSpinBox->blockSignals(b);
}

void MinMaxThresholdWidget::SetLimitsVisible(bool b)
{
  this->limitsGroupBox->setVisible(b);
}

void MinMaxThresholdWidget::SetLimits(double min, double max)
{
  this->BlockAllSignals(true);

  this->minSpinBox->setMinimum(min);
  this->maxSpinBox->setMinimum(min);
  this->thresholdSpinBox->setMinimum(min);

  this->minSpinBox->setMaximum(max);
  this->maxSpinBox->setMaximum(max);
  this->thresholdSpinBox->setMaximum(max);

  this->minLimitSpinBox->setValue(min);
  this->maxLimitSpinBox->setValue(max);

  this->BlockAllSignals(false);
}

void MinMaxThresholdWidget::SetIntensities(double currentMin, double currentMax, double currentThreshold)
{
  this->BlockAllSignals(true);

  int sliderVal = ConvertSpinBoxValueToSliderValue(currentMin, this->minSpinBox->minimum(), this->minSpinBox->maximum(), m_SliderMin, m_SliderMax);
  this->m_PreviousMinimumIntensity = currentMin;
  this->minSpinBox->setValue(currentMin);
  this->minSlider->setValue(sliderVal);

  sliderVal = ConvertSpinBoxValueToSliderValue(currentMax, this->maxSpinBox->minimum(), this->maxSpinBox->maximum(), m_SliderMin, m_SliderMax);
  this->m_PreviousMaximumIntensity = currentMax;
  this->maxSpinBox->setValue(currentMax);
  this->maxSlider->setValue(sliderVal);

  sliderVal = ConvertSpinBoxValueToSliderValue(currentThreshold, this->thresholdSpinBox->minimum(), this->thresholdSpinBox->maximum(), m_SliderMin, m_SliderMax);
  this->m_PreviousMinimumThreshold = currentThreshold;
  this->thresholdSpinBox->setValue(currentThreshold);
  this->thresholdSlider->setValue(sliderVal);

  this->BlockAllSignals(false);
}

void MinMaxThresholdWidget::SetMinimumIntensity(double d)
{
  int sliderValue = ConvertSpinBoxValueToSliderValue(d, this->minSpinBox->minimum(), this->minSpinBox->maximum(), m_SliderMin, m_SliderMax);

  this->minSpinBox->blockSignals(true);
  this->minSlider->blockSignals(true);

  this->minSpinBox->setValue(d);
  this->minSlider->setValue(sliderValue);

  this->minSpinBox->blockSignals(false);
  this->minSlider->blockSignals(false);

}

void MinMaxThresholdWidget::SetMaximumIntensity(double d)
{
  int sliderValue = ConvertSpinBoxValueToSliderValue(d, this->maxSpinBox->minimum(), this->maxSpinBox->maximum(), m_SliderMin, m_SliderMax);

  this->maxSpinBox->blockSignals(true);
  this->maxSlider->blockSignals(true);

  this->maxSpinBox->setValue(d);
  this->maxSlider->setValue(sliderValue);

  this->maxSpinBox->blockSignals(false);
  this->maxSlider->blockSignals(false);

}

void MinMaxThresholdWidget::SetMinimumThreshold(double d)
{

  int sliderValue = ConvertSpinBoxValueToSliderValue(d, this->thresholdSpinBox->minimum(), this->thresholdSpinBox->maximum(), m_SliderMin, m_SliderMax);

  this->thresholdSpinBox->blockSignals(true);
  this->thresholdSlider->blockSignals(true);

  this->thresholdSpinBox->setValue(d);
  this->thresholdSlider->setValue(sliderValue);

  this->thresholdSpinBox->blockSignals(false);
  this->thresholdSlider->blockSignals(false);

}

double MinMaxThresholdWidget::GetTolerance()
{
  return fabs((double)((this->minSpinBox->maximum() - this->minSpinBox->minimum())/1000.0));
}

void MinMaxThresholdWidget::OnMinimumIntensitySpinBoxChanged(double d)
{
  double tolerance = this->GetTolerance();
  int sliderValue = ConvertSpinBoxValueToSliderValue(d, this->minSpinBox->minimum(), this->minSpinBox->maximum(), m_SliderMin, m_SliderMax);

  this->minSlider->blockSignals(true);
  this->minSlider->setValue(sliderValue);
  this->minSlider->blockSignals(false);

  double currentMax = this->maxSpinBox->value();
  int currentMaxSliderValue = 0;

  if (d >= currentMax - tolerance)
  {
    currentMax = d + tolerance;
    currentMaxSliderValue = (int)ConvertSpinBoxValueToSliderValue(currentMax, this->maxSpinBox->minimum(), this->maxSpinBox->maximum(), m_SliderMin, m_SliderMax);

    this->maxSlider->setValue(currentMaxSliderValue);
  }

  emit MinimumIntensityChanged(m_PreviousMinimumIntensity, d);
  m_PreviousMinimumIntensity = d;
}

void MinMaxThresholdWidget::OnMaximumIntensitySpinBoxChanged(double d)
{
  double tolerance = this->GetTolerance();
  int sliderValue = ConvertSpinBoxValueToSliderValue(d, this->maxSpinBox->minimum(), this->maxSpinBox->maximum(), m_SliderMin, m_SliderMax);

  this->maxSlider->blockSignals(true);
  this->maxSlider->setValue(sliderValue);
  this->maxSlider->blockSignals(false);

  double currentMin = this->minSpinBox->value();
  int currentMinSliderValue = 0;

  if (d <= currentMin + tolerance)
  {
    currentMin = d - tolerance;
    currentMinSliderValue = (int)ConvertSpinBoxValueToSliderValue(currentMin, this->minSpinBox->minimum(), this->minSpinBox->maximum(), m_SliderMin, m_SliderMax);
    this->minSlider->setValue(currentMinSliderValue);
  }

  emit MaximumIntensityChanged(m_PreviousMaximumIntensity, d);
  m_PreviousMaximumIntensity = d;
}

void MinMaxThresholdWidget::OnMinimumThresholdSpinBoxChanged(double d)
{
  int sliderValue = ConvertSpinBoxValueToSliderValue(d, this->thresholdSpinBox->minimum(), this->thresholdSpinBox->maximum(), m_SliderMin, m_SliderMax);
  this->thresholdSlider->blockSignals(true);
  this->thresholdSlider->setValue(sliderValue);
  this->thresholdSlider->blockSignals(true);
  emit MinimumThresholdChanged(m_PreviousMinimumThreshold, d);
  m_PreviousMinimumThreshold = d;
}


void MinMaxThresholdWidget::OnMinimumIntensitySliderChanged(int i)
{
  double tolerance = this->GetTolerance();
  double spinBoxValue = ConvertSliderValueToSpinBoxValue(i, this->minSpinBox->minimum(), this->minSpinBox->maximum(), m_SliderMin, m_SliderMax);

  double currentMax = this->maxSpinBox->value();
  int currentMaxSliderValue = 0;

  if (spinBoxValue >= currentMax - tolerance)
  {
    currentMax = spinBoxValue + tolerance;
    currentMaxSliderValue = (int)ConvertSpinBoxValueToSliderValue(currentMax, this->maxSpinBox->minimum(), this->maxSpinBox->maximum(), m_SliderMin, m_SliderMax);
    this->maxSlider->setValue(currentMaxSliderValue);
  }

  this->minSpinBox->blockSignals(true);
  this->minSpinBox->setValue(spinBoxValue);
  this->minSpinBox->blockSignals(false);
  emit MinimumIntensityChanged(m_PreviousMinimumIntensity, spinBoxValue);
  m_PreviousMinimumIntensity = spinBoxValue;
}

void MinMaxThresholdWidget::OnMaximumIntensitySliderChanged(int i)
{
  double tolerance = this->GetTolerance();
  double spinBoxValue = ConvertSliderValueToSpinBoxValue(i, this->maxSpinBox->minimum(), this->maxSpinBox->maximum(), m_SliderMin, m_SliderMax);

  double currentMin = this->minSpinBox->value();
  int currentMinSliderValue = 0;
  if (spinBoxValue <= currentMin + tolerance)
  {
    currentMin = spinBoxValue - tolerance;
    currentMinSliderValue = (int)ConvertSpinBoxValueToSliderValue(currentMin, this->minSpinBox->minimum(), this->minSpinBox->maximum(), m_SliderMin, m_SliderMax);
    this->minSlider->setValue(currentMinSliderValue);
  }

  this->maxSpinBox->blockSignals(true);
  this->maxSpinBox->setValue(spinBoxValue);
  this->maxSpinBox->blockSignals(false);
  emit MaximumIntensityChanged(m_PreviousMaximumIntensity, spinBoxValue);
  m_PreviousMaximumIntensity = spinBoxValue;
}

void MinMaxThresholdWidget::OnMinimumThresholdSliderChanged(int i)
{
  double spinBoxValue = ConvertSliderValueToSpinBoxValue(i, this->thresholdSpinBox->minimum(), this->thresholdSpinBox->maximum(), m_SliderMin, m_SliderMax);
  this->thresholdSpinBox->blockSignals(true);
  this->thresholdSpinBox->setValue(spinBoxValue);
  this->thresholdSpinBox->blockSignals(false);
  emit MinimumThresholdChanged(m_PreviousMinimumThreshold, spinBoxValue);
  m_PreviousMinimumThreshold = spinBoxValue;
}

void MinMaxThresholdWidget::OnMoreButtonPressed()
{
  if (this->limitsGroupBox->isVisible())
  {
    this->SetLimitsVisible(false);
  }
  else
  {
    this->SetLimitsVisible(true);
  }
}

void MinMaxThresholdWidget::OnMinLimitSpinBoxChanged(double d)
{
  this->minSpinBox->setMinimum(d);
  this->minSpinBox->setValue(d);

  this->thresholdSpinBox->setMinimum(d);
  this->thresholdSpinBox->setValue(d);

  this->maxSpinBox->setMinimum(d);
}

void MinMaxThresholdWidget::OnMaxLimitSpinBoxChanged(double d)
{
  this->maxSpinBox->setMaximum(d);
  this->maxSpinBox->setValue(d);

  this->thresholdSpinBox->setMaximum(d);
  this->minSpinBox->setMaximum(d);
}

#endif
