/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASGENERALSEGMENTORVIEWCONTROLSWIDGET_CPP
#define MIDASGENERALSEGMENTORVIEWCONTROLSWIDGET_CPP

#include "MIDASGeneralSegmentorViewControlsWidget.h"

//-----------------------------------------------------------------------------
MIDASGeneralSegmentorViewControlsWidget::MIDASGeneralSegmentorViewControlsWidget(QWidget *parent)
: QWidget(parent)
{
  if (parent)
  {
    setupUi(parent);
  }
}


//-----------------------------------------------------------------------------
MIDASGeneralSegmentorViewControlsWidget::~MIDASGeneralSegmentorViewControlsWidget()
{

}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::setupUi(QWidget* parent)
{
  Ui_MIDASGeneralSegmentorViewControls::setupUi(parent);

  this->SetEnableAllWidgets(false);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetEnableThresholdingCheckbox(bool enabled)
{
  this->m_ThresholdCheckBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetEnableThresholdingWidgets(bool enabled)
{
  this->m_Prop3DButton->setEnabled(enabled);
  this->m_PropDownButton->setEnabled(enabled);
  this->m_PropUpButton->setEnabled(enabled);
  this->m_ThresholdApplyButton->setEnabled(enabled);
  this->m_ThresholdLowerLabel->setEnabled(enabled);
  this->m_ThresholdLowerSliderWidget->setEnabled(enabled);
  this->m_ThresholdUpperLabel->setEnabled(enabled);
  this->m_ThresholdUpperSliderWidget->setEnabled(enabled);
  this->m_ThresholdSeedMaxLabel->setEnabled(enabled);
  this->m_ThresholdSeedMaxValue->setEnabled(enabled);
  this->m_ThresholdSeedMinLabel->setEnabled(enabled);
  this->m_ThresholdSeedMinValue->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetEnableOKCancelResetWidgets(bool enabled)
{
  m_OKButton->setEnabled(enabled);
  m_ResetButton->setEnabled(enabled);
  m_CancelButton->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetEnableAllWidgets(bool enabled)
{
  this->SetEnableThresholdingCheckbox(enabled);
  this->SetEnableThresholdingWidgets(enabled);
  this->SetEnableOKCancelResetWidgets(enabled);
  m_RetainMarksCheckBox->setEnabled(enabled);
  m_SeePriorCheckBox->setEnabled(enabled);
  m_SeeNextCheckBox->setEnabled(enabled);
  m_SeeImageCheckBox->setEnabled(enabled);
  m_CleanButton->setEnabled(enabled);
  m_WipeButton->setEnabled(enabled);
  m_WipePlusButton->setEnabled(enabled);
  m_WipeMinusButton->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetLowerAndUpperIntensityRanges(double lower, double upper)
{
  m_ThresholdLowerSliderWidget->setMinimum(lower);
  m_ThresholdLowerSliderWidget->setMaximum(upper);
  m_ThresholdUpperSliderWidget->setMinimum(lower);
  m_ThresholdUpperSliderWidget->setMaximum(upper);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetSeedMinAndMaxValues(double min, double max)
{
  QString minText;
  QString maxText;

  minText.sprintf("%.2f", min);
  maxText.sprintf("%.2f", max);

  m_ThresholdSeedMinValue->setText(minText);
  m_ThresholdSeedMaxValue->setText(maxText);
}
#endif
