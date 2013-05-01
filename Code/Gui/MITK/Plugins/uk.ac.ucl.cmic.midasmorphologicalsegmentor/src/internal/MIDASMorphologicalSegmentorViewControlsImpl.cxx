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

  m_ThresholdingAxialCutoffSlider->setSingleStep(1.0);
  m_ThresholdingAxialCutoffSlider->setDecimals(0);
  // Trick alert!
  // So that the width of the spinbox is equal to the other spinboxes:
  m_ThresholdingAxialCutoffSlider->setMaximum(100.0);

  m_ErosionsUpperThresholdSlider->setTracking(false);
  m_ErosionsNumberOfErosionsSlider->setMinimum(0.0);
  m_ErosionsNumberOfErosionsSlider->setMaximum(6.0);
  m_ErosionsNumberOfErosionsSlider->setValue(0.0);
  m_ErosionsNumberOfErosionsSlider->setSingleStep(1.0);
  m_ErosionsNumberOfErosionsSlider->setDecimals(0);
  m_ErosionsNumberOfErosionsSlider->setTickInterval(1.0);
//  m_ErosionsNumberOfErosionsSlider->setTickPosition(QSlider::TicksBelow);

  m_DilationsLowerThresholdSlider->setMinimum(0);
  m_DilationsLowerThresholdSlider->setMaximum(100);
  m_DilationsLowerThresholdSlider->setValue(60);
  m_DilationsLowerThresholdSlider->setTickInterval(1);

  m_DilationsUpperThresholdSlider->setMinimum(100);
  m_DilationsUpperThresholdSlider->setMaximum(300);
  m_DilationsUpperThresholdSlider->setValue(160);
  m_DilationsUpperThresholdSlider->setTickInterval(1);

  m_DilationsNumberOfDilationsSlider->setMinimum(0.0);
  m_DilationsNumberOfDilationsSlider->setMaximum(10.0);
  m_DilationsNumberOfDilationsSlider->setValue(0.0);
  m_DilationsNumberOfDilationsSlider->setSingleStep(1.0);
  m_DilationsNumberOfDilationsSlider->setDecimals(0);
  m_DilationsNumberOfDilationsSlider->setTickInterval(1.0);
//  m_DilationsNumberOfDilationsSlider->setTickPosition(QSlider::TicksBelow);

  m_RethresholdingBoxSizeSlider->setSingleStep(1.0);
  m_RethresholdingBoxSizeSlider->setDecimals(0);
  m_RethresholdingBoxSizeSlider->setTickInterval(1.0);
  m_RethresholdingBoxSizeSlider->setMinimum(0.0);
  m_RethresholdingBoxSizeSlider->setMaximum(10.0);
  m_RethresholdingBoxSizeSlider->setValue(0.0);
//  m_RethresholdingBoxSizeSlider->setTickPosition(QSlider::TicksBelow);

  connect(m_ThresholdingLowerThresholdSlider, SIGNAL(valueChanged(double)), this, SLOT(OnThresholdLowerValueChanged(double)));
  connect(m_ThresholdingUpperThresholdSlider, SIGNAL(valueChanged(double)), this, SLOT(OnThresholdUpperValueChanged(double)));
  connect(m_ThresholdingAxialCutoffSlider, SIGNAL(valueChanged(double)), this, SLOT(OnAxialCuttoffSliderChanged()));
  connect(m_BackButton, SIGNAL(clicked()), this, SLOT(OnBackButtonClicked()));
  connect(m_NextButton, SIGNAL(clicked()), this, SLOT(OnNextButtonClicked()));
  connect(m_ErosionsUpperThresholdSlider, SIGNAL(valueChanged(double)), this, SLOT(OnErosionsUpperThresholdChanged()));
  connect(m_ErosionsNumberOfErosionsSlider, SIGNAL(valueChanged(double)), this, SLOT(OnErosionsSliderChanged()));
  connect(m_DilationsNumberOfDilationsSlider, SIGNAL(valueChanged(double)), this, SLOT(OnDilationsSliderChanged()));
  connect(m_RethresholdingBoxSizeSlider, SIGNAL(valueChanged(double)), this, SLOT(OnRethresholdingSliderChanged()));
  connect(m_CancelButton, SIGNAL(clicked()), this, SIGNAL(CancelButtonClicked()));
  connect(m_RestartButton, SIGNAL(clicked()), this, SLOT(OnRestartButtonClicked()));

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
  m_ErosionsNumberOfErosionsSlider->setEnabled(enable);
  m_ErosionsUpperThresholdSlider->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab3Dilations(bool enable)
{
  m_DilationsLowerThresholdSlider->setEnabled(enable);
  m_DilationsUpperThresholdSlider->setEnabled(enable);
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
  m_CancelButton->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableRestartButton(bool enable)
{
    m_RestartButton->setEnabled(enable);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableByTabNumber(int i)
{
  if (i == 0)
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
  else if (i == 1)
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
  }
  else if (i == 2)
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
  }
  else if (i == 3)
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
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EnableControls(bool b)
{
  if (b)
  {
    this->EnableByTabNumber(this->GetTabNumber());
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
void MIDASMorphologicalSegmentorViewControlsImpl::SetControlsByImageData(double lowestValue, double highestValue, int numberAxialSlices)
{
  this->blockSignals(true);

  double stepSize = 1;
  double pageSize = 10;

  if (fabs((double)(highestValue - lowestValue)) < 50)
  {
    stepSize = (highestValue-lowestValue)/100.0;
    pageSize = (highestValue-lowestValue)/10.0;
  }
  m_ThresholdingLowerThresholdSlider->setMinimum(lowestValue);
  m_ThresholdingLowerThresholdSlider->setMaximum(highestValue);
  m_ThresholdingLowerThresholdSlider->setSingleStep(stepSize);
  m_ThresholdingLowerThresholdSlider->setPageStep(pageSize);
  m_ThresholdingLowerThresholdSlider->setValue(lowestValue);
  m_ThresholdingUpperThresholdSlider->setMinimum(lowestValue);
  m_ThresholdingUpperThresholdSlider->setMaximum(highestValue);
  m_ThresholdingUpperThresholdSlider->setValue(lowestValue); // Intentionally set to lowest values, as this is what MIDAS does.
  m_ThresholdingUpperThresholdSlider->setSingleStep(stepSize);
  m_ThresholdingUpperThresholdSlider->setPageStep(pageSize);
  m_ThresholdingAxialCutoffSlider->setMinimum(0);
  m_ThresholdingAxialCutoffSlider->setMaximum(numberAxialSlices - 1);

  m_ErosionsUpperThresholdSlider->setSingleStep(stepSize);
  m_ErosionsUpperThresholdSlider->setPageStep(pageSize);

  m_DilationsLowerThresholdSlider->setSingleStep(1);  // this is a percentage.
  m_DilationsLowerThresholdSlider->setPageStep(10); // this is a percentage.
  m_DilationsUpperThresholdSlider->setSingleStep(1);  // this is a percentage.
  m_DilationsUpperThresholdSlider->setPageStep(10); // this is a percentage.

  this->blockSignals(false);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::SetControlsByParameterValues(MorphologicalSegmentorPipelineParams &params)
{
  this->blockSignals(true);

  m_ThresholdingLowerThresholdSlider->setValue(params.m_LowerIntensityThreshold);
  m_ThresholdingUpperThresholdSlider->setValue(params.m_UpperIntensityThreshold);
  m_ThresholdingAxialCutoffSlider->setValue(params.m_AxialCutoffSlice);
  m_ErosionsUpperThresholdSlider->setValue(params.m_UpperErosionsThreshold);
  m_ErosionsNumberOfErosionsSlider->setValue(params.m_NumberOfErosions);
  m_DilationsLowerThresholdSlider->setValue(params.m_LowerPercentageThresholdForDilations);
  m_DilationsUpperThresholdSlider->setValue(params.m_UpperPercentageThresholdForDilations);
  m_DilationsNumberOfDilationsSlider->setValue(params.m_NumberOfDilations);
  m_RethresholdingBoxSizeSlider->setValue(params.m_BoxSize);

  this->blockSignals(false);

  this->SetTabNumber(params.m_Stage);
}


//-----------------------------------------------------------------------------
int MIDASMorphologicalSegmentorViewControlsImpl::GetTabNumber()
{
  return m_TabWidget->currentIndex();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::SetTabNumber(int i)
{
  if (i == 0)
  {
  }
  if (i == 1)
  {
    m_ErosionsUpperThresholdSlider->setMinimum(m_ThresholdingLowerThresholdSlider->value());
    m_ErosionsUpperThresholdSlider->setMaximum(m_ThresholdingUpperThresholdSlider->value());
    m_ErosionsUpperThresholdSlider->setValue(m_ThresholdingUpperThresholdSlider->value());
  }

  this->EnableByTabNumber(i);

  m_TabWidget->setCurrentIndex(i);
  emit TabChanged(i);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::EmitThresholdingValues()
{
  emit ThresholdingValuesChanged(
         m_ThresholdingLowerThresholdSlider->value(),
         m_ThresholdingUpperThresholdSlider->value(),
         m_ThresholdingAxialCutoffSlider->value()
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
         m_DilationsLowerThresholdSlider->value(),
         m_DilationsUpperThresholdSlider->value(),
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
void MIDASMorphologicalSegmentorViewControlsImpl::OnThresholdLowerValueChanged(double d)
{
  if (d >= m_ThresholdingUpperThresholdSlider->value())
  {
    m_ThresholdingUpperThresholdSlider->blockSignals(true);
    m_ThresholdingUpperThresholdSlider->setValue(d + m_ThresholdingUpperThresholdSlider->tickInterval());
    m_ThresholdingUpperThresholdSlider->blockSignals(false);
  }
  this->EmitThresholdingValues();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnThresholdUpperValueChanged(double d)
{
  if (d <= m_ThresholdingLowerThresholdSlider->value())
  {
    m_ThresholdingLowerThresholdSlider->blockSignals(true);
    m_ThresholdingLowerThresholdSlider->setValue(d - m_ThresholdingLowerThresholdSlider->tickInterval());
    m_ThresholdingLowerThresholdSlider->blockSignals(false);
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
  this->SetTabNumber(tabNumber - 1);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewControlsImpl::OnNextButtonClicked()
{
  int tabNumber = this->GetTabNumber();
  if (tabNumber < 3)
  {
    this->SetTabNumber(tabNumber + 1);
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
  emit RestartButtonClicked();
}

#endif
