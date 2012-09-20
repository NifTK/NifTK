/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-02 14:50:50 +0000 (Wed, 02 Nov 2011) $
 Revision          : $Revision: 7660 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MIDASMORPHOLOGICALSEGMENTORCONTROLSIMPL_CPP
#define MIDASMORPHOLOGICALSEGMENTORCONTROLSIMPL_CPP

#include "MIDASMorphologicalSegmentorViewControlsImpl.h"
#include <iostream>
#include <cmath>

MIDASMorphologicalSegmentorViewControlsImpl::MIDASMorphologicalSegmentorViewControlsImpl()
{
  this->setupUi(this);
}

MIDASMorphologicalSegmentorViewControlsImpl::~MIDASMorphologicalSegmentorViewControlsImpl()
{

}

void MIDASMorphologicalSegmentorViewControlsImpl::setupUi(QWidget* parent)
{
  Ui_MIDASMorphologicalSegmentorViewControls::setupUi(parent);

  m_ErosionsUpperThresholdSlider->setTracking(false);
  m_ErosionsNumberOfErosionsSlider->setValue(0);
  m_ErosionsNumberOfErosionsLineEdit->setText(QString::number(0));
  m_DilationsLowerThresholdSlider->setMinimum(0);
  m_DilationsLowerThresholdSlider->setMaximum(100);
  m_DilationsLowerThresholdSlider->setValue(60);
  m_DilationsLowerThresholdSlider->setTickInterval(1);
  m_DilationsUpperThresholdSlider->setMinimum(100);
  m_DilationsUpperThresholdSlider->setMaximum(300);
  m_DilationsUpperThresholdSlider->setValue(160);
  m_DilationsUpperThresholdSlider->setTickInterval(1);

  m_segmentationDialogButtonBox->button(QDialogButtonBox::Ok)->setObjectName(QString::fromUtf8("OkButton"));
  m_segmentationDialogButtonBox->button(QDialogButtonBox::Cancel)->setObjectName(QString::fromUtf8("CancelButton"));

  // TODO: Decide how we will provide help.
  m_segmentationDialogHelpPushButton->setVisible(false);

  connect(m_ThresholdingLowerThresholdSlider, SIGNAL(valueChanged(double)), this, SLOT(OnThresholdLowerValueChanged(double)));
  connect(m_ThresholdingUpperThresholdSlider, SIGNAL(valueChanged(double)), this, SLOT(OnThresholdUpperValueChanged(double)));
  connect(m_ThresholdingAxialCutoffSlider, SIGNAL(valueChanged(int)), this, SLOT(OnAxialCuttoffSliderChanged(int)));
  connect(m_ThresholdingAxialCutoffSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnAxialCuttoffSpinBoxChanged(int)));
  connect(m_ThresholdingAcceptButton, SIGNAL(pressed()), this, SLOT(OnThresholdAcceptButtonClicked()));
  connect(m_ErosionsUpperThresholdSlider, SIGNAL(valueChanged(double)), this, SLOT(OnErosionsUpperThresholdChanged(double)));
  connect(m_ErosionsNumberOfErosionsSlider, SIGNAL(sliderMoved(int)), this, SLOT(OnErosionsSliderMoved(int)));
  connect(m_ErosionsNumberOfErosionsSlider, SIGNAL(valueChanged(int)), this, SLOT(OnErosionsSliderChanged(int)));
  connect(m_ErosionsResetButton, SIGNAL(pressed()), this, SLOT(OnErosionsResetButtonClicked()));
  connect(m_ErosionsAcceptButton, SIGNAL(pressed()), this, SLOT(OnErosionsAcceptButtonClicked()));
  connect(m_DilationsNumberOfDilationsSlider, SIGNAL(sliderMoved(int)), this, SLOT(OnDilationsSliderMoved(int)));
  connect(m_DilationsNumberOfDilationsSlider, SIGNAL(valueChanged(int)), this, SLOT(OnDilationsSliderChanged(int)));
  connect(m_DilationsResetButton, SIGNAL(pressed()), this, SLOT(OnDilationsResetButtonClicked()));
  connect(m_DilationsAcceptButton, SIGNAL(pressed()), this, SLOT(OnDilationsAcceptButtonClicked()));
  connect(m_RethresholdingHorizontalSlider, SIGNAL(sliderMoved(int)), this, SLOT(OnRethresholdingSliderMoved(int)));
  connect(m_RethresholdingHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(OnRethresholdingSliderChanged(int)));
  connect(m_RethresholdingResetButton, SIGNAL(pressed()), this, SLOT(OnRethresholdingResetButtonClicked()));
  connect(m_segmentationDialogButtonBox, SIGNAL(clicked(QAbstractButton*)), this, SLOT(OnOKCancelClicked(QAbstractButton*)));
  connect(m_segmentationDialogClearPushButton, SIGNAL(clicked()), this, SLOT(OnClearButtonClicked()));

  this->EnableControls(false);
}

void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab1Thresholding(bool enable)
{
  m_ThresholdingAcceptButton->setEnabled(enable);
  m_ThresholdingAxialCutoffSlider->setEnabled(enable);
  m_ThresholdingAxialCutoffSpinBox->setEnabled(enable);
  m_ThresholdingLowerThresholdSlider->setEnabled(enable);
  m_ThresholdingUpperThresholdSlider->setEnabled(enable);
}

void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab2Erosions(bool enable)
{
  m_ErosionsAcceptButton->setEnabled(enable);
  m_ErosionsNumberOfErosionsSlider->setEnabled(enable);
  m_ErosionsNumberOfErosionsLineEdit->setEnabled(enable);
  m_ErosionsResetButton->setEnabled(enable);
  m_ErosionsUpperThresholdSlider->setEnabled(enable);
}

void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab3Dilations(bool enable)
{
  m_DilationsAcceptButton->setEnabled(enable);
  m_DilationsResetButton->setEnabled(enable);
  m_DilationsLowerThresholdSlider->setEnabled(enable);
  m_DilationsNumberOfDilationsSlider->setEnabled(enable);
  m_DilationsNumberOfDilationsLineEdit->setEnabled(enable);
  m_DilationsUpperThresholdSlider->setEnabled(enable);
}

void MIDASMorphologicalSegmentorViewControlsImpl::EnableTab4ReThresholding(bool enable)
{
  m_RethresholdingHorizontalSlider->setEnabled(enable);
  m_RethresholdingResetButton->setEnabled(enable);
  m_RethresholdingLineEdit->setEnabled(enable);
}

void MIDASMorphologicalSegmentorViewControlsImpl::EnableOKButton(bool enable)
{
  m_segmentationDialogButtonBox->button(QDialogButtonBox::Ok)->setEnabled(enable);
}

void MIDASMorphologicalSegmentorViewControlsImpl::EnableCancelButton(bool enable)
{
  m_segmentationDialogButtonBox->button(QDialogButtonBox::Cancel)->setEnabled(enable);
}

void MIDASMorphologicalSegmentorViewControlsImpl::EnableResetButton(bool enable)
{
    m_segmentationDialogClearPushButton->setEnabled(enable);
}

void MIDASMorphologicalSegmentorViewControlsImpl::EnableByTabNumber(int i)
{
  if (i == 0)
  {
    this->EnableTab1Thresholding(true);
    this->EnableTab2Erosions(false);
    this->EnableTab3Dilations(false);
    this->EnableTab4ReThresholding(false);
    this->EnableOKButton(false);
    this->EnableCancelButton(true);
    this->EnableResetButton(false);
  }
  else if (i == 1)
  {
    this->EnableTab1Thresholding(false);
    this->EnableTab2Erosions(true);
    this->EnableTab3Dilations(false);
    this->EnableTab4ReThresholding(false);
    this->EnableOKButton(false);
    this->EnableCancelButton(true);
    this->EnableResetButton(true);
  }
  else if (i == 2)
  {
    this->EnableTab1Thresholding(false);
    this->EnableTab2Erosions(false);
    this->EnableTab3Dilations(true);
    this->EnableTab4ReThresholding(false);
    this->EnableOKButton(false);
    this->EnableCancelButton(true);
    this->EnableResetButton(true);
  }
  else if (i == 3)
  {
    this->EnableTab1Thresholding(false);
    this->EnableTab2Erosions(false);
    this->EnableTab3Dilations(false);
    this->EnableTab4ReThresholding(true);
    this->EnableOKButton(true);
    this->EnableCancelButton(true);
    this->EnableResetButton(true);
  }
}

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
    this->EnableOKButton(false);
    this->EnableCancelButton(false);
    this->EnableResetButton(false);
  }
}

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
  m_ThresholdingAxialCutoffSlider->setMaximum(numberAxialSlices-1);
  m_ThresholdingAxialCutoffSpinBox->setMinimum(0);
  m_ThresholdingAxialCutoffSpinBox->setMaximum(numberAxialSlices-1);

  m_ErosionsUpperThresholdSlider->setSingleStep(stepSize);
  m_ErosionsUpperThresholdSlider->setPageStep(pageSize);

  m_DilationsLowerThresholdSlider->setSingleStep(1);  // this is a percentage.
  m_DilationsLowerThresholdSlider->setPageStep(10); // this is a percentage.
  m_DilationsUpperThresholdSlider->setSingleStep(1);  // this is a percentage.
  m_DilationsUpperThresholdSlider->setPageStep(10); // this is a percentage.

  this->blockSignals(false);
}

void MIDASMorphologicalSegmentorViewControlsImpl::SetControlsByParameterValues(MorphologicalSegmentorPipelineParams &params)
{
  this->blockSignals(true);

  m_ThresholdingLowerThresholdSlider->setValue(params.m_LowerIntensityThreshold);
  m_ThresholdingUpperThresholdSlider->setValue(params.m_UpperIntensityThreshold);
  m_ThresholdingAxialCutoffSpinBox->setValue(params.m_AxialCutoffSlice);
  m_ThresholdingAxialCutoffSlider->setValue(params.m_AxialCutoffSlice);
  m_ErosionsUpperThresholdSlider->setValue(params.m_UpperErosionsThreshold);
  m_ErosionsNumberOfErosionsSlider->setValue(params.m_NumberOfErosions);
  m_ErosionsNumberOfErosionsLineEdit->setText(QString::number(params.m_NumberOfErosions));
  m_DilationsLowerThresholdSlider->setValue(params.m_LowerPercentageThresholdForDilations);
  m_DilationsUpperThresholdSlider->setValue(params.m_UpperPercentageThresholdForDilations);
  m_DilationsNumberOfDilationsLineEdit->setText(QString::number(params.m_NumberOfDilations));
  m_DilationsNumberOfDilationsSlider->setValue(params.m_NumberOfDilations);
  m_RethresholdingHorizontalSlider->setValue(params.m_BoxSize);
  m_RethresholdingLineEdit->setText(QString::number(params.m_BoxSize));

  this->blockSignals(false);

  this->SetTabNumber(params.m_Stage);
}

int MIDASMorphologicalSegmentorViewControlsImpl::GetTabNumber()
{
  return m_TabWidget->currentIndex();
}

void MIDASMorphologicalSegmentorViewControlsImpl::SetTabNumber(int i)
{
  if (i == 1)
  {
    this->m_ErosionsUpperThresholdSlider->setMinimum(this->m_ThresholdingLowerThresholdSlider->value());
    this->m_ErosionsUpperThresholdSlider->setMaximum(this->m_ThresholdingUpperThresholdSlider->value());
    this->m_ErosionsUpperThresholdSlider->setValue(this->m_ThresholdingUpperThresholdSlider->value());
  }

  this->EnableByTabNumber(i);

  m_TabWidget->setCurrentIndex(i);
  emit TabChanged(i);
}

void MIDASMorphologicalSegmentorViewControlsImpl::EmitThresholdingValues()
{
  emit ThresholdingValuesChanged(
         m_ThresholdingLowerThresholdSlider->value(),
         m_ThresholdingUpperThresholdSlider->value(),
         m_ThresholdingAxialCutoffSpinBox->value()
       );
}

void MIDASMorphologicalSegmentorViewControlsImpl::EmitErosionValues()
{
  emit ErosionsValuesChanged(
         m_ErosionsUpperThresholdSlider->value(),
         m_ErosionsNumberOfErosionsLineEdit->text().toInt()
       );
}

void MIDASMorphologicalSegmentorViewControlsImpl::EmitDilationValues()
{
  emit DilationValuesChanged(
         m_DilationsLowerThresholdSlider->value(),
         m_DilationsUpperThresholdSlider->value(),
         m_DilationsNumberOfDilationsLineEdit->text().toInt()
       );
}

void MIDASMorphologicalSegmentorViewControlsImpl::EmitRethresholdingValues()
{
  emit RethresholdingValuesChanged(
         m_RethresholdingLineEdit->text().toInt()
      );
}

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

void MIDASMorphologicalSegmentorViewControlsImpl::OnAxialCuttoffSliderChanged(int i)
{
  this->m_ThresholdingAxialCutoffSpinBox->blockSignals(true);
  this->m_ThresholdingAxialCutoffSpinBox->setValue(i);
  this->m_ThresholdingAxialCutoffSpinBox->blockSignals(false);
  this->EmitThresholdingValues();
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnAxialCuttoffSpinBoxChanged(int i)
{
  this->m_ThresholdingAxialCutoffSlider->blockSignals(true);
  this->m_ThresholdingAxialCutoffSlider->setValue(i);
  this->m_ThresholdingAxialCutoffSlider->blockSignals(false);
  this->EmitThresholdingValues();
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnThresholdAcceptButtonClicked()
{
  this->SetTabNumber(1);
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnErosionsSliderMoved(int i)
{
  this->m_ErosionsNumberOfErosionsLineEdit->blockSignals(true);
  this->m_ErosionsNumberOfErosionsLineEdit->setText(QString::number(i));
  this->m_ErosionsNumberOfErosionsLineEdit->blockSignals(false);
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnErosionsSliderChanged(int i)
{
  this->m_ErosionsNumberOfErosionsLineEdit->blockSignals(true);
  this->m_ErosionsNumberOfErosionsLineEdit->setText(QString::number(i));
  this->m_ErosionsNumberOfErosionsLineEdit->blockSignals(false);
  this->EmitErosionValues();
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnErosionsUpperThresholdChanged(double)
{
  this->EmitErosionValues();
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnErosionsAcceptButtonClicked()
{
  this->SetTabNumber(2);
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnErosionsResetButtonClicked()
{
  this->SetTabNumber(0);
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnDilationsSliderMoved(int i)
{
  this->m_DilationsNumberOfDilationsLineEdit->blockSignals(true);
  this->m_DilationsNumberOfDilationsLineEdit->setText(QString::number(i));
  this->m_DilationsNumberOfDilationsLineEdit->blockSignals(false);
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnDilationsSliderChanged(int i)
{
  this->m_DilationsNumberOfDilationsLineEdit->blockSignals(true);
  this->m_DilationsNumberOfDilationsLineEdit->setText(QString::number(i));
  this->m_DilationsNumberOfDilationsLineEdit->blockSignals(false);
  this->EmitDilationValues();
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnDilationsAcceptButtonClicked()
{
  this->SetTabNumber(3);
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnDilationsResetButtonClicked()
{
  this->SetTabNumber(1);
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnRethresholdingSliderMoved(int i)
{
  this->m_RethresholdingLineEdit->blockSignals(true);
  this->m_RethresholdingLineEdit->setText(QString::number(i));
  this->m_RethresholdingLineEdit->blockSignals(false);
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnRethresholdingSliderChanged(int i)
{
  this->m_RethresholdingLineEdit->blockSignals(true);
  this->m_RethresholdingLineEdit->setText(QString::number(i));
  this->m_RethresholdingLineEdit->blockSignals(false);
  this->EmitRethresholdingValues();
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnRethresholdingResetButtonClicked()
{
  this->SetTabNumber(2);
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnOKCancelClicked(QAbstractButton *button)
{
  if (button->objectName() == QString("OkButton"))
  {
    emit OKButtonClicked();
  }
  else if (button->objectName() == QString("CancelButton"))
  {
    emit CancelButtonClicked();
  }
}

void MIDASMorphologicalSegmentorViewControlsImpl::OnClearButtonClicked()
{
  emit ClearButtonClicked();
}

#endif
