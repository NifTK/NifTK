/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorControls.h"

#include <niftkToolSelectorWidget.h>


//-----------------------------------------------------------------------------
niftkGeneralSegmentorControls::niftkGeneralSegmentorControls(QWidget *parent)
: niftkBaseSegmentorControls(parent)
{
  QGridLayout* layout = new QGridLayout(parent);
  layout->setContentsMargins(6, 6, 6, 0);
  layout->setSpacing(3);

  QWidget* containerForControlsWidget = new QWidget(parent);
  containerForControlsWidget->setContentsMargins(0, 0, 0, 0);

  this->setupUi(containerForControlsWidget);

  layout->addWidget(m_ContainerForSelectorWidget, 0, 0);
  layout->addWidget(m_ContainerForToolWidget, 1, 0);
  layout->addWidget(containerForControlsWidget, 2, 0);

  layout->setRowStretch(0, 0);
  layout->setRowStretch(1, 1);
  layout->setRowStretch(2, 0);

  m_ToolSelectorWidget->SetDisplayedToolGroups("Seed Draw Poly");

  this->SetThresholdingCheckboxEnabled(false);
  this->SetThresholdingWidgetsEnabled(false);
}


//-----------------------------------------------------------------------------
niftkGeneralSegmentorControls::~niftkGeneralSegmentorControls()
{
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorControls::setupUi(QWidget* parent)
{
  Ui_niftkGeneralSegmentorWidget::setupUi(parent);

  m_ThresholdsSlider->layout()->setSpacing(2);

  this->SetAllWidgetsEnabled(false);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorControls::SetThresholdingCheckboxEnabled(bool enabled)
{
  m_ThresholdingCheckBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorControls::SetThresholdingWidgetsEnabled(bool enabled)
{
  m_ThresholdingGroupBox->setEnabled(enabled);
//  m_ThresholdingGroupBox->setVisible(enabled);

  m_SeedMinLabel->setEnabled(enabled);
  m_SeedMinValue->setEnabled(enabled);
  m_SeedMaxLabel->setEnabled(enabled);
  m_SeedMaxValue->setEnabled(enabled);

  m_ThresholdsSlider->setEnabled(enabled);

  m_PropUpButton->setEnabled(enabled);
  m_PropDownButton->setEnabled(enabled);
  m_Prop3DButton->setEnabled(enabled);
  m_ThresholdApplyButton->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorControls::SetOKCancelResetWidgetsEnabled(bool enabled)
{
  m_OKButton->setEnabled(enabled);
  m_CancelButton->setEnabled(enabled);
  m_ResetButton->setEnabled(enabled);
  m_RestartButton->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorControls::SetAllWidgetsEnabled(bool enabled)
{
  this->SetThresholdingCheckboxEnabled(enabled);
  this->SetThresholdingWidgetsEnabled(enabled);
  this->SetOKCancelResetWidgetsEnabled(enabled);
  m_RetainMarksCheckBox->setEnabled(enabled);
  m_SeePriorCheckBox->setEnabled(enabled);
  m_SeeNextCheckBox->setEnabled(enabled);
  m_CleanButton->setEnabled(enabled);
  m_WipeButton->setEnabled(enabled);
  m_WipePlusButton->setEnabled(enabled);
  m_WipeMinusButton->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorControls::SetLowerAndUpperIntensityRanges(double lower, double upper)
{
  m_ThresholdsSlider->setMinimum(lower);
  m_ThresholdsSlider->setMaximum(upper);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorControls::SetSeedMinAndMaxValues(double min, double max)
{
  QString minText;
  QString maxText;

  minText.sprintf("%.2f", min);
  maxText.sprintf("%.2f", max);

  m_SeedMinValue->setText(minText);
  m_SeedMaxValue->setText(maxText);
}
