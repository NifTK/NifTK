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

  this->SetAllWidgetsEnabled(false);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetThresholdingCheckboxEnabled(bool enabled)
{
  m_ThresholdingCheckBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetThresholdingWidgetsEnabled(bool enabled)
{
  m_ThresholdingGroupBox->setEnabled(enabled);
//  m_ThresholdingGroupBox->setVisible(enabled);

  m_SeedMinLabel->setEnabled(enabled);
  m_SeedMinValue->setEnabled(enabled);
  m_SeedMaxLabel->setEnabled(enabled);
  m_SeedMaxValue->setEnabled(enabled);

  m_LowerThresholdLabel->setEnabled(enabled);
  m_LowerThresholdSliderWidget->setEnabled(enabled);
  m_UpperThresholdLabel->setEnabled(enabled);
  m_UpperThresholdSliderWidget->setEnabled(enabled);

  m_PropUpButton->setEnabled(enabled);
  m_PropDownButton->setEnabled(enabled);
  m_Prop3DButton->setEnabled(enabled);
  m_ThresholdApplyButton->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetOKCancelResetWidgetsEnabled(bool enabled)
{
  m_OKButton->setEnabled(enabled);
  m_ResetButton->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetAllWidgetsEnabled(bool enabled)
{
  this->SetThresholdingCheckboxEnabled(enabled);
  this->SetThresholdingWidgetsEnabled(enabled);
  this->SetOKCancelResetWidgetsEnabled(enabled);
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
  m_LowerThresholdSliderWidget->setMinimum(lower);
  m_LowerThresholdSliderWidget->setMaximum(upper);
  m_UpperThresholdSliderWidget->setMinimum(lower);
  m_UpperThresholdSliderWidget->setMaximum(upper);
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewControlsWidget::SetSeedMinAndMaxValues(double min, double max)
{
  QString minText;
  QString maxText;

  minText.sprintf("%.2f", min);
  maxText.sprintf("%.2f", max);

  m_SeedMinValue->setText(minText);
  m_SeedMaxValue->setText(maxText);
}
#endif
