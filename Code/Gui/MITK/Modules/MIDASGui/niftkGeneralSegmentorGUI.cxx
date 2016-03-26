/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorGUI.h"

#include <niftkToolSelectorWidget.h>


//-----------------------------------------------------------------------------
niftkGeneralSegmentorGUI::niftkGeneralSegmentorGUI(QWidget *parent)
: niftkBaseSegmentorGUI(parent)
{
  QGridLayout* layout = new QGridLayout(parent);
  layout->setContentsMargins(6, 6, 6, 0);
  layout->setSpacing(3);

  QWidget* containerForControlsWidget = new QWidget(parent);
  containerForControlsWidget->setContentsMargins(0, 0, 0, 0);

  Ui_niftkGeneralSegmentorWidget::setupUi(containerForControlsWidget);

  layout->addWidget(m_ContainerForSelectorWidget, 0, 0);
  layout->addWidget(m_ContainerForToolWidget, 1, 0);
  layout->addWidget(containerForControlsWidget, 2, 0);

  layout->setRowStretch(0, 0);
  layout->setRowStretch(1, 1);
  layout->setRowStretch(2, 0);

  m_ToolSelectorWidget->SetDisplayedToolGroups("Seed Draw Poly");

  m_ThresholdsSlider->layout()->setSpacing(2);

  this->SetThresholdingCheckBoxEnabled(false);
  this->SetThresholdingWidgetsEnabled(false);
  this->SetAllWidgetsEnabled(false);

  this->connect(m_CleanButton, SIGNAL(clicked()), SIGNAL(CleanButtonClicked()));
  this->connect(m_WipeButton, SIGNAL(clicked()), SIGNAL(WipeButtonClicked()));
  this->connect(m_WipePlusButton, SIGNAL(clicked()), SIGNAL(WipePlusButtonClicked()));
  this->connect(m_WipeMinusButton, SIGNAL(clicked()), SIGNAL(WipeMinusButtonClicked()));
  this->connect(m_PropUpButton, SIGNAL(clicked()), SIGNAL(PropagateUpButtonClicked()));
  this->connect(m_PropDownButton, SIGNAL(clicked()), SIGNAL(PropagateDownButtonClicked()));
  this->connect(m_Prop3DButton, SIGNAL(clicked()), SIGNAL(Propagate3DButtonClicked()));
  this->connect(m_OKButton, SIGNAL(clicked()), SIGNAL(OKButtonClicked()));
  this->connect(m_CancelButton, SIGNAL(clicked()), SIGNAL(CancelButtonClicked()));
  this->connect(m_RestartButton, SIGNAL(clicked()), SIGNAL(RestartButtonClicked()));
  this->connect(m_ResetButton, SIGNAL(clicked()), SIGNAL(ResetButtonClicked()));
  this->connect(m_ThresholdApplyButton, SIGNAL(clicked()), SIGNAL(ThresholdApplyButtonClicked()));
  this->connect(m_ThresholdingCheckBox, SIGNAL(toggled(bool)), SIGNAL(ThresholdingCheckBoxToggled(bool)));
  this->connect(m_SeePriorCheckBox, SIGNAL(toggled(bool)), SIGNAL(SeePriorCheckBoxToggled(bool)));
  this->connect(m_SeeNextCheckBox, SIGNAL(toggled(bool)), SIGNAL(SeeNextCheckBoxToggled(bool)));
  this->connect(m_ThresholdsSlider, SIGNAL(minimumValueChanged(double)), SIGNAL(ThresholdValueChanged()));
  this->connect(m_ThresholdsSlider, SIGNAL(maximumValueChanged(double)), SIGNAL(ThresholdValueChanged()));
}


//-----------------------------------------------------------------------------
niftkGeneralSegmentorGUI::~niftkGeneralSegmentorGUI()
{
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorGUI::SetThresholdingCheckBoxEnabled(bool enabled)
{
  m_ThresholdingCheckBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorGUI::SetThresholdingWidgetsEnabled(bool enabled)
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
bool niftkGeneralSegmentorGUI::IsThresholdingCheckBoxChecked() const
{
  return m_ThresholdingCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorGUI::SetThresholdingCheckBoxChecked(bool checked)
{
  bool wasBlocked = m_ThresholdingCheckBox->blockSignals(true);
  m_ThresholdingCheckBox->setChecked(checked);
  m_ThresholdingCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorGUI::IsSeePriorCheckBoxChecked() const
{
  return m_SeePriorCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorGUI::SetSeePriorCheckBoxChecked(bool checked)
{
  m_SeePriorCheckBox->setChecked(checked);
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorGUI::IsSeeNextCheckBoxChecked() const
{
  return m_SeeNextCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorGUI::SetSeeNextCheckBoxChecked(bool checked)
{
  m_SeeNextCheckBox->setChecked(checked);
}


//-----------------------------------------------------------------------------
bool niftkGeneralSegmentorGUI::IsRetainMarksCheckBoxChecked() const
{
  return m_RetainMarksCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorGUI::SetRetainMarksCheckBoxChecked(bool checked)
{
  m_RetainMarksCheckBox->setChecked(checked);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorGUI::SetOKCancelResetWidgetsEnabled(bool enabled)
{
  m_OKButton->setEnabled(enabled);
  m_CancelButton->setEnabled(enabled);
  m_ResetButton->setEnabled(enabled);
  m_RestartButton->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorGUI::SetAllWidgetsEnabled(bool enabled)
{
  this->SetThresholdingCheckBoxEnabled(enabled);
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
void niftkGeneralSegmentorGUI::SetLowerAndUpperIntensityRanges(double lower, double upper)
{
  m_ThresholdsSlider->setMinimum(lower);
  m_ThresholdsSlider->setMaximum(upper);
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorGUI::SetSeedMinAndMaxValues(double min, double max)
{
  QString minText;
  QString maxText;

  minText.sprintf("%.2f", min);
  maxText.sprintf("%.2f", max);

  m_SeedMinValue->setText(minText);
  m_SeedMaxValue->setText(maxText);
}


//-----------------------------------------------------------------------------
double niftkGeneralSegmentorGUI::GetLowerThreshold() const
{
  return m_ThresholdsSlider->minimumValue();
}


//-----------------------------------------------------------------------------
double niftkGeneralSegmentorGUI::GetUpperThreshold() const
{
  return m_ThresholdsSlider->maximumValue();
}
