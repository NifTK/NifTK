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
#include <ctkSliderWidget.h>

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

  m_ThresholdingLowerThresholdSlider->layout()->setSpacing(2);
  m_ThresholdingLowerThresholdSlider->setSpinBoxAlignment(Qt::AlignRight);
  m_ThresholdingLowerThresholdSlider->setDecimals(0);
  m_ThresholdingLowerThresholdSlider->setMinimum(0.0);
  m_ThresholdingLowerThresholdSlider->setMaximum(0.0);

  m_ThresholdingUpperThresholdSlider->layout()->setSpacing(2);
  m_ThresholdingUpperThresholdSlider->setSpinBoxAlignment(Qt::AlignRight);
  m_ThresholdingUpperThresholdSlider->setDecimals(0);
  m_ThresholdingUpperThresholdSlider->setMinimum(0.0);
  m_ThresholdingUpperThresholdSlider->setMaximum(0.0);

  m_ThresholdingAxialCutoffSlider->layout()->setSpacing(2);
  m_ThresholdingAxialCutoffSlider->setSpinBoxAlignment(Qt::AlignRight);
  m_ThresholdingAxialCutoffSlider->setSingleStep(1.0);
  m_ThresholdingAxialCutoffSlider->setPageStep(2.0);
  m_ThresholdingAxialCutoffSlider->setDecimals(0);
  // Trick alert!
  // So that the width of the spinbox is equal to the other spinboxes:
  m_ThresholdingAxialCutoffSlider->setMaximum(100.0);

  m_ErosionsUpperThresholdSlider->setTracking(false);
  m_ErosionsUpperThresholdSlider->layout()->setSpacing(2);
  m_ErosionsUpperThresholdSlider->setSpinBoxAlignment(Qt::AlignRight);
  m_ErosionsIterationsSlider->layout()->setSpacing(2);
  m_ErosionsIterationsSlider->setSpinBoxAlignment(Qt::AlignRight);
  m_ErosionsIterationsSlider->setMinimum(0.0);
  m_ErosionsIterationsSlider->setMaximum(6.0);
  m_ErosionsIterationsSlider->setValue(0.0);
  m_ErosionsIterationsSlider->setSingleStep(1.0);
  m_ErosionsIterationsSlider->setDecimals(0);
  m_ErosionsIterationsSlider->setTickInterval(1.0);
  m_ErosionsIterationsSlider->setTickPosition(QSlider::TicksBelow);

  m_DilationsLowerThresholdSlider->layout()->setSpacing(2);
  m_DilationsLowerThresholdSlider->setSpinBoxAlignment(Qt::AlignRight);
  m_DilationsLowerThresholdSlider->setDecimals(0);
  m_DilationsLowerThresholdSlider->setMinimum(0);
  m_DilationsLowerThresholdSlider->setMaximum(300);
  m_DilationsLowerThresholdSlider->setValue(60);
  m_DilationsLowerThresholdSlider->setTickInterval(1.0);
  m_DilationsLowerThresholdSlider->setSuffix("%");

  m_DilationsUpperThresholdSlider->layout()->setSpacing(2);
  m_DilationsUpperThresholdSlider->setSpinBoxAlignment(Qt::AlignRight);
  m_DilationsUpperThresholdSlider->setDecimals(0);
  m_DilationsUpperThresholdSlider->setMinimum(0);
  m_DilationsUpperThresholdSlider->setMaximum(300);
  m_DilationsUpperThresholdSlider->setValue(160);
  m_DilationsUpperThresholdSlider->setTickInterval(1.0);
  m_DilationsUpperThresholdSlider->setSuffix("%");

  m_DilationsIterationsSlider->layout()->setSpacing(2);
  m_DilationsIterationsSlider->setSpinBoxAlignment(Qt::AlignRight);
  m_DilationsIterationsSlider->setMinimum(0.0);
  m_DilationsIterationsSlider->setMaximum(10.0);
  m_DilationsIterationsSlider->setValue(0.0);
  m_DilationsIterationsSlider->setSingleStep(1.0);
  m_DilationsIterationsSlider->setDecimals(0);
  m_DilationsIterationsSlider->setSynchronizeSiblings(ctkSliderWidget::SynchronizeWidth);
  m_DilationsIterationsSlider->setTickInterval(1.0);
  m_DilationsIterationsSlider->setTickPosition(QSlider::TicksBelow);

  m_RethresholdingBoxSizeSlider->layout()->setSpacing(2);
  m_RethresholdingBoxSizeSlider->setSpinBoxAlignment(Qt::AlignRight);
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

  this->connect(m_ThresholdingLowerThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnThresholdLowerValueChanged()));
  this->connect(m_ThresholdingUpperThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnThresholdUpperValueChanged()));
  this->connect(m_ThresholdingAxialCutoffSlider, SIGNAL(valueChanged(double)), SLOT(OnAxialCuttoffSliderChanged()));
  this->connect(m_BackButton, SIGNAL(clicked()), SLOT(OnBackButtonClicked()));
  this->connect(m_NextButton, SIGNAL(clicked()), SLOT(OnNextButtonClicked()));
  this->connect(m_ErosionsUpperThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnErosionsUpperThresholdChanged()));
  this->connect(m_ErosionsIterationsSlider, SIGNAL(valueChanged(double)), SLOT(OnErosionsIterationsChanged()));
  this->connect(m_DilationsLowerThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnDilationsLowerThresholdChanged()));
  this->connect(m_DilationsUpperThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnDilationsUpperThresholdChanged()));
  this->connect(m_DilationsIterationsSlider, SIGNAL(valueChanged(double)), SLOT(OnDilationsIterationsChanged()));
  this->connect(m_RethresholdingBoxSizeSlider, SIGNAL(valueChanged(double)), SLOT(OnRethresholdingSliderChanged()));
  this->connect(m_RestartButton, SIGNAL(clicked()), SLOT(OnRestartButtonClicked()));

  this->EnableControls(false);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab1Thresholding(bool enable)
{
  m_ThresholdingAxialCutoffSlider->setEnabled(enable);
  m_ThresholdingLowerThresholdSlider->setEnabled(enable);
  m_ThresholdingUpperThresholdSlider->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab2Erosions(bool enable)
{
  m_ErosionsIterationsSlider->setEnabled(enable);
  m_ErosionsUpperThresholdSlider->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab3Dilations(bool enable)
{
  m_DilationsLowerThresholdSlider->setEnabled(enable);
  m_DilationsUpperThresholdSlider->setEnabled(enable);
  m_DilationsIterationsSlider->setEnabled(enable);
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
void MIDASMorphologicalSegmentorViewControlsImpl::EnableControls(bool enabled)
{
  if (enabled)
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
  bool wasBlocked = this->blockSignals(true);

  double stepSize = 1;
  double pageSize = 10;

  if (fabs((double)(highestValue - lowestValue)) < 50)
  {
    stepSize = (highestValue - lowestValue) / 100.0;
    pageSize = (highestValue - lowestValue) / 10.0;
  }
  m_ThresholdingLowerThresholdSlider->setMinimum(lowestValue);
  m_ThresholdingLowerThresholdSlider->setMaximum(highestValue);
  m_ThresholdingLowerThresholdSlider->setSingleStep(stepSize);
  m_ThresholdingLowerThresholdSlider->setPageStep(pageSize);
  m_ThresholdingLowerThresholdSlider->setValue(lowestValue);

  m_ThresholdingUpperThresholdSlider->setMinimum(lowestValue);
  m_ThresholdingUpperThresholdSlider->setMaximum(highestValue);
  m_ThresholdingUpperThresholdSlider->setSingleStep(stepSize);
  m_ThresholdingUpperThresholdSlider->setPageStep(pageSize);
  m_ThresholdingUpperThresholdSlider->setValue(lowestValue); // Intentionally set to lowest values, as this is what MIDAS does.

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

  m_ErosionsIterationsSlider->setSingleStep(1.0);
  m_ErosionsIterationsSlider->setPageStep(1.0);

  m_DilationsLowerThresholdSlider->setSingleStep(1.0);  // this is a percentage.
  m_DilationsLowerThresholdSlider->setPageStep(10.0);   // this is a percentage.
  m_DilationsUpperThresholdSlider->setSingleStep(1.0);  // this is a percentage.
  m_DilationsUpperThresholdSlider->setPageStep(10.0);   // this is a percentage.

  m_DilationsIterationsSlider->setSingleStep(1.0);
  m_DilationsIterationsSlider->setPageStep(1.0);

  this->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::SetControlsByParameterValues(MorphologicalSegmentorPipelineParams &params)
{
  bool wasBlocked = this->blockSignals(true);

  m_ThresholdingLowerThresholdSlider->setValue(params.m_LowerIntensityThreshold);
  m_ThresholdingUpperThresholdSlider->setValue(params.m_UpperIntensityThreshold);
  m_ThresholdingAxialCutoffSlider->setValue(params.m_AxialCutoffSlice);
  m_ErosionsUpperThresholdSlider->setValue(params.m_UpperErosionsThreshold);
  m_ErosionsIterationsSlider->setValue(params.m_NumberOfErosions);
  m_DilationsLowerThresholdSlider->setValue(params.m_LowerPercentageThresholdForDilations);
  m_DilationsUpperThresholdSlider->setValue(params.m_UpperPercentageThresholdForDilations);
  m_DilationsIterationsSlider->setValue(params.m_NumberOfDilations);
  m_RethresholdingBoxSizeSlider->setValue(params.m_BoxSize);

  this->blockSignals(wasBlocked);

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
    m_ErosionsUpperThresholdSlider->setMinimum(m_ThresholdingLowerThresholdSlider->value());
    m_ErosionsUpperThresholdSlider->setMaximum(m_ThresholdingUpperThresholdSlider->value());
    m_ErosionsUpperThresholdSlider->setValue(m_ThresholdingUpperThresholdSlider->value());
  }

  this->EnableByTabIndex(tabIndex);

  m_TabWidget->setCurrentIndex(tabIndex);
  emit TabChanged(tabIndex);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EmitThresholdingValues()
{
  emit ThresholdingValuesChanged(
         m_ThresholdingLowerThresholdSlider->value(),
         m_ThresholdingUpperThresholdSlider->value(),
         static_cast<int>(m_ThresholdingAxialCutoffSlider->value())
       );
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EmitErosionValues()
{
  emit ErosionsValuesChanged(
         m_ErosionsUpperThresholdSlider->value(),
         static_cast<int>(m_ErosionsIterationsSlider->value())
       );
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EmitDilationValues()
{
  emit DilationValuesChanged(
         m_DilationsLowerThresholdSlider->value(),
         m_DilationsUpperThresholdSlider->value(),
         static_cast<int>(m_DilationsIterationsSlider->value())
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
  double lowerValue = m_ThresholdingLowerThresholdSlider->value();
  double upperValue = m_ThresholdingUpperThresholdSlider->value();
  if (lowerValue >= upperValue)
  {
    bool wasBlocked = m_ThresholdingUpperThresholdSlider->blockSignals(true);
    m_ThresholdingUpperThresholdSlider->setValue(lowerValue + m_ThresholdingUpperThresholdSlider->singleStep());
    m_ThresholdingUpperThresholdSlider->blockSignals(wasBlocked);
  }
  this->EmitThresholdingValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnThresholdUpperValueChanged()
{
  double lowerValue = m_ThresholdingLowerThresholdSlider->value();
  double upperValue = m_ThresholdingUpperThresholdSlider->value();
  if (lowerValue >= upperValue)
  {
    bool wasBlocked = m_ThresholdingLowerThresholdSlider->blockSignals(true);
    m_ThresholdingLowerThresholdSlider->setValue(upperValue - m_ThresholdingLowerThresholdSlider->singleStep());
    m_ThresholdingLowerThresholdSlider->blockSignals(wasBlocked);
  }
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
void MIDASMorphologicalSegmentorViewControlsImpl::OnErosionsIterationsChanged()
{
  this->EmitErosionValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnErosionsUpperThresholdChanged()
{
  this->EmitErosionValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnDilationsLowerThresholdChanged()
{
  double lowerValue = m_DilationsLowerThresholdSlider->value();
  double upperValue = m_DilationsUpperThresholdSlider->value();
  if (lowerValue >= upperValue)
  {
    bool wasBlocked = m_DilationsUpperThresholdSlider->blockSignals(true);
    m_DilationsUpperThresholdSlider->setValue(lowerValue + m_DilationsUpperThresholdSlider->singleStep());
    m_DilationsUpperThresholdSlider->blockSignals(wasBlocked);
  }
  this->EmitDilationValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnDilationsUpperThresholdChanged()
{
  double lowerValue = m_DilationsLowerThresholdSlider->value();
  double upperValue = m_DilationsUpperThresholdSlider->value();
  if (lowerValue >= upperValue)
  {
    bool wasBlocked = m_DilationsLowerThresholdSlider->blockSignals(true);
    m_DilationsLowerThresholdSlider->setValue(upperValue - m_DilationsLowerThresholdSlider->singleStep());
    m_DilationsLowerThresholdSlider->blockSignals(wasBlocked);
  }
  this->EmitDilationValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnDilationsIterationsChanged()
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
