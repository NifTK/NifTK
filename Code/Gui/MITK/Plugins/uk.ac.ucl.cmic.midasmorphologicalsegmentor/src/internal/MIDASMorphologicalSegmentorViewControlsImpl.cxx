/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASMORPHOLOGICALSEGMENTORCONTROLSIMPL_CPP
#define MIDASMORPHOLOGICALSEGMENTORCONTROLSIMPL_CPP

#include "MIDASMorphologicalSegmentorViewControlsImpl.h"
#include <iostream>
#include <cmath>
#include <ctkDoubleSlider.h>
#include <ctkRangeWidget.h>

//-----------------------------------------------------------------------------
MIDASMorphologicalSegmentorViewControlsImpl::MIDASMorphologicalSegmentorViewControlsImpl()
{
  this->setupUi(this);
}


//-----------------------------------------------------------------------------
MIDASMorphologicalSegmentorViewControlsImpl::~MIDASMorphologicalSegmentorViewControlsImpl()
{
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::setupUi(QWidget* parent)
{
  Ui_MIDASMorphologicalSegmentorViewControls::setupUi(parent);

  m_ThresholdingThresholdsSlider->layout()->setSpacing(2);

  m_ThresholdingAxialCutoffSlider->layout()->setSpacing(2);
  m_ThresholdingAxialCutoffSlider->setSingleStep(1.0);
  m_ThresholdingAxialCutoffSlider->setDecimals(0);
  // Trick alert!
  // So that the width of the spinbox is equal to the other spinboxes:
  m_ThresholdingAxialCutoffSlider->setMaximum(100.0);

  m_ErosionsUpperThresholdSlider->setTracking(false);
  m_ErosionsUpperThresholdSlider->layout()->setSpacing(2);
  m_ErosionsNumberOfErosionsSlider->layout()->setSpacing(2);
  m_ErosionsNumberOfErosionsSlider->setMinimum(0.0);
  m_ErosionsNumberOfErosionsSlider->setMaximum(6.0);
  m_ErosionsNumberOfErosionsSlider->setValue(0.0);
  m_ErosionsNumberOfErosionsSlider->setSingleStep(1.0);
  m_ErosionsNumberOfErosionsSlider->setDecimals(0);
  m_ErosionsNumberOfErosionsSlider->setTickInterval(1.0);
  m_ErosionsNumberOfErosionsSlider->setTickPosition(QSlider::TicksBelow);

  m_DilationsThresholdsSlider->layout()->setSpacing(2);
  m_DilationsThresholdsSlider->setMinimum(0);
  m_DilationsThresholdsSlider->setMaximum(300);
  m_DilationsThresholdsSlider->setMinimumValue(60);
  m_DilationsThresholdsSlider->setMaximumValue(160);
  m_DilationsThresholdsSlider->setTickInterval(1.0);

  m_DilationsNumberOfDilationsSlider->layout()->setSpacing(2);
  m_DilationsNumberOfDilationsSlider->setMinimum(0.0);
  m_DilationsNumberOfDilationsSlider->setMaximum(10.0);
  m_DilationsNumberOfDilationsSlider->setValue(0.0);
  m_DilationsNumberOfDilationsSlider->setSingleStep(1.0);
  m_DilationsNumberOfDilationsSlider->setDecimals(0);
  m_DilationsNumberOfDilationsSlider->setTickInterval(1.0);
  m_DilationsNumberOfDilationsSlider->setTickPosition(QSlider::TicksBelow);

  m_RethresholdingBoxSizeSlider->layout()->setSpacing(2);
  m_RethresholdingBoxSizeSlider->setSingleStep(1.0);
  m_RethresholdingBoxSizeSlider->setDecimals(0);
  m_RethresholdingBoxSizeSlider->setTickInterval(1.0);
  m_RethresholdingBoxSizeSlider->setMinimum(0.0);
  m_RethresholdingBoxSizeSlider->setMaximum(10.0);
  m_RethresholdingBoxSizeSlider->setValue(0.0);
  m_RethresholdingBoxSizeSlider->setTickPosition(QSlider::TicksBelow);

  m_TabWidget->setTabEnabled(1, false);
  m_TabWidget->setTabEnabled(2, false);
  m_TabWidget->setTabEnabled(3, false);

  this->connect(m_ThresholdingThresholdsSlider, SIGNAL(minimumValueChanged(double)), SLOT(OnThresholdLowerValueChanged()));
  this->connect(m_ThresholdingThresholdsSlider, SIGNAL(maximumValueChanged(double)), SLOT(OnThresholdUpperValueChanged()));
  this->connect(m_ThresholdingAxialCutoffSlider, SIGNAL(valueChanged(double)), SLOT(OnAxialCuttoffSliderChanged()));
  this->connect(m_BackButton, SIGNAL(clicked()), SLOT(OnBackButtonClicked()));
  this->connect(m_NextButton, SIGNAL(clicked()), SLOT(OnNextButtonClicked()));
  this->connect(m_ErosionsUpperThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnErosionsUpperThresholdChanged()));
  this->connect(m_ErosionsNumberOfErosionsSlider, SIGNAL(valueChanged(double)), SLOT(OnErosionsSliderChanged()));
  this->connect(m_DilationsNumberOfDilationsSlider, SIGNAL(valueChanged(double)), SLOT(OnDilationsSliderChanged()));
  this->connect(m_RethresholdingBoxSizeSlider, SIGNAL(valueChanged(double)), SLOT(OnRethresholdingSliderChanged()));
  this->connect(m_RestartButton, SIGNAL(clicked()), SLOT(OnRestartButtonClicked()));

  this->EnableControls(false);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab1Thresholding(bool enable)
{
  m_ThresholdingAxialCutoffSlider->setEnabled(enable);
  m_ThresholdingThresholdsSlider->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab2Erosions(bool enable)
{
  m_ErosionsNumberOfErosionsSlider->setEnabled(enable);
  m_ErosionsUpperThresholdSlider->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab3Dilations(bool enable)
{
  m_DilationsThresholdsSlider->setEnabled(enable);
  m_DilationsNumberOfDilationsSlider->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab4ReThresholding(bool enable)
{
  m_RethresholdingBoxSizeSlider->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableCancelButton(bool enable)
{
//  m_CancelButton->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableRestartButton(bool enable)
{
    m_RestartButton->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableByTabIndex(int tabIndex)
{
  if (tabIndex == 0)
  {
    this->EnableTab1Thresholding(true);
    this->EnableTab2Erosions(false);
    this->EnableTab3Dilations(false);
    this->EnableTab4ReThresholding(false);
    m_BackButton->setEnabled(false);
    m_NextButton->setEnabled(true);
    m_NextButton->setText("Next >");
    this->EnableCancelButton(true);
    this->EnableRestartButton(false);
  }
  else if (tabIndex == 1)
  {
    this->EnableTab1Thresholding(false);
    this->EnableTab2Erosions(true);
    this->EnableTab3Dilations(false);
    this->EnableTab4ReThresholding(false);
    m_BackButton->setEnabled(true);
    m_NextButton->setEnabled(true);
    m_NextButton->setText("Next >");
    this->EnableCancelButton(true);
    this->EnableRestartButton(true);
    m_TabWidget->setTabEnabled(tabIndex, true);
  }
  else if (tabIndex == 2)
  {
    this->EnableTab1Thresholding(false);
    this->EnableTab2Erosions(false);
    this->EnableTab3Dilations(true);
    this->EnableTab4ReThresholding(false);
    m_BackButton->setEnabled(true);
    m_NextButton->setEnabled(true);
    m_NextButton->setText("Next >");
    this->EnableCancelButton(true);
    this->EnableRestartButton(true);
    m_TabWidget->setTabEnabled(tabIndex, true);
  }
  else if (tabIndex == 3)
  {
    this->EnableTab1Thresholding(false);
    this->EnableTab2Erosions(false);
    this->EnableTab3Dilations(false);
    this->EnableTab4ReThresholding(true);
    m_BackButton->setEnabled(true);
    m_NextButton->setEnabled(true);
    m_NextButton->setText("Finish");
    this->EnableCancelButton(true);
    this->EnableRestartButton(true);
    m_TabWidget->setTabEnabled(tabIndex, true);
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableControls(bool b)
{
  if (b)
  {
    this->EnableByTabIndex(this->GetTabNumber());
  }
  else
  {
    this->EnableTab1Thresholding(false);
    this->EnableTab2Erosions(false);
    this->EnableTab3Dilations(false);
    this->EnableTab4ReThresholding(false);
    this->EnableCancelButton(false);
    m_BackButton->setEnabled(false);
    m_NextButton->setText("Next >");
    m_NextButton->setEnabled(false);
    this->EnableRestartButton(false);
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::SetControlsByImageData(double lowestValue, double highestValue, int numberOfAxialSlices, int upDirection)
{
  this->blockSignals(true);

  double stepSize = 1;
  double pageSize = 10;

  if (fabs((double)(highestValue - lowestValue)) < 50)
  {
    stepSize = (highestValue - lowestValue) / 100.0;
    pageSize = (highestValue - lowestValue) / 10.0;
  }
  m_ThresholdingThresholdsSlider->setMinimum(lowestValue);
  m_ThresholdingThresholdsSlider->setMaximum(highestValue);
  m_ThresholdingThresholdsSlider->setSingleStep(stepSize);
  // Not implemented for ctkRangeWidget.
//  m_ThresholdingThresholdsSlider->setPageStep(pageSize);
  m_ThresholdingThresholdsSlider->setMinimumValue(lowestValue);
  m_ThresholdingThresholdsSlider->setMaximumValue(lowestValue); // Intentionally set to lowest values, as this is what MIDAS does.
  m_ThresholdingAxialCutoffSlider->setMinimum(0);
  m_ThresholdingAxialCutoffSlider->setMaximum(numberOfAxialSlices - 1);
  if (upDirection > 0)
  {
    m_ThresholdingAxialCutoffSlider->setInvertedAppearance(false);
    m_ThresholdingAxialCutoffSlider->setInvertedControls(false);
    m_ThresholdingAxialCutoffSlider->setValue(0);
  }
  else
  {
    m_ThresholdingAxialCutoffSlider->setInvertedAppearance(true);
    m_ThresholdingAxialCutoffSlider->setInvertedControls(true);
    m_ThresholdingAxialCutoffSlider->setValue(numberOfAxialSlices - 1);
  }

  m_ErosionsUpperThresholdSlider->setSingleStep(stepSize);
  m_ErosionsUpperThresholdSlider->setPageStep(pageSize);

  m_DilationsThresholdsSlider->setSingleStep(1);  // this is a percentage.
  // Not implemented for ctkRangeWidget.
//  m_DilationsThresholdsSlider->setPageStep(10); // this is a percentage.

  this->blockSignals(false);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::SetControlsByParameterValues(MorphologicalSegmentorPipelineParams &params)
{
  this->blockSignals(true);

  m_ThresholdingThresholdsSlider->setMinimumValue(params.m_LowerIntensityThreshold);
  m_ThresholdingThresholdsSlider->setMaximumValue(params.m_UpperIntensityThreshold);
  m_ThresholdingAxialCutoffSlider->setValue(params.m_AxialCutoffSlice);
  m_ErosionsUpperThresholdSlider->setValue(params.m_UpperErosionsThreshold);
  m_ErosionsNumberOfErosionsSlider->setValue(params.m_NumberOfErosions);
  m_DilationsThresholdsSlider->setMinimumValue(params.m_LowerPercentageThresholdForDilations);
  m_DilationsThresholdsSlider->setMaximumValue(params.m_UpperPercentageThresholdForDilations);
  m_DilationsNumberOfDilationsSlider->setValue(params.m_NumberOfDilations);
  m_RethresholdingBoxSizeSlider->setValue(params.m_BoxSize);

  this->blockSignals(false);

  this->SetTabIndex(params.m_Stage);
}


//-----------------------------------------------------------------------------
int MIDASMorphologicalSegmentorViewControlsImpl::GetTabNumber()
{
  return m_TabWidget->currentIndex();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::SetTabIndex(int tabIndex)
{
  if (tabIndex == 1)
  {
    m_ErosionsUpperThresholdSlider->setMinimum(m_ThresholdingThresholdsSlider->minimumValue());
    m_ErosionsUpperThresholdSlider->setMaximum(m_ThresholdingThresholdsSlider->maximumValue());
    m_ErosionsUpperThresholdSlider->setValue(m_ThresholdingThresholdsSlider->maximumValue());
  }

  this->EnableByTabIndex(tabIndex);

  m_TabWidget->setCurrentIndex(tabIndex);
  emit TabChanged(tabIndex);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EmitThresholdingValues()
{
  emit ThresholdingValuesChanged(
         m_ThresholdingThresholdsSlider->minimumValue(),
         m_ThresholdingThresholdsSlider->maximumValue(),
         static_cast<int>(m_ThresholdingAxialCutoffSlider->value())
       );
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EmitErosionValues()
{
  emit ErosionsValuesChanged(
         m_ErosionsUpperThresholdSlider->value(),
         static_cast<int>(m_ErosionsNumberOfErosionsSlider->value())
       );
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EmitDilationValues()
{
  emit DilationValuesChanged(
         m_DilationsThresholdsSlider->minimumValue(),
         m_DilationsThresholdsSlider->maximumValue(),
         static_cast<int>(m_DilationsNumberOfDilationsSlider->value())
       );
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EmitRethresholdingValues()
{
  emit RethresholdingValuesChanged(
         static_cast<int>(m_RethresholdingBoxSizeSlider->value())
      );
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnThresholdLowerValueChanged()
{
  this->EmitThresholdingValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnThresholdUpperValueChanged()
{
  this->EmitThresholdingValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnAxialCuttoffSliderChanged()
{
  this->EmitThresholdingValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnBackButtonClicked()
{
  int tabNumber = this->GetTabNumber();
  this->SetTabIndex(tabNumber - 1);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnNextButtonClicked()
{
  int tabNumber = this->GetTabNumber();
  if (tabNumber < 3)
  {
    this->SetTabIndex(tabNumber + 1);
  }
  else
  {
    emit this->OKButtonClicked();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnErosionsSliderChanged()
{
  this->EmitErosionValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnErosionsUpperThresholdChanged()
{
  this->EmitErosionValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnDilationsSliderChanged()
{
  this->EmitDilationValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnRethresholdingSliderChanged()
{
  this->EmitRethresholdingValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnRestartButtonClicked()
{
  m_TabWidget->setTabEnabled(1, false);
  m_TabWidget->setTabEnabled(2, false);
  m_TabWidget->setTabEnabled(3, false);
  emit RestartButtonClicked();
}

#endif
