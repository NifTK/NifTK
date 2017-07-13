/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMorphologicalSegmentorGUI.h"

#include "niftkToolSelectorWidget.h"

namespace niftk
{

//-----------------------------------------------------------------------------
MorphologicalSegmentorGUI::MorphologicalSegmentorGUI(QWidget* parent)
  : BaseSegmentorGUI(parent)
{
  QGridLayout* layout = new QGridLayout(parent);
  layout->setContentsMargins(6, 6, 6, 0);
  layout->setSpacing(3);

  QWidget* containerForControlsWidget = new QWidget(parent);
  containerForControlsWidget->setContentsMargins(0, 0, 0, 0);

  Ui_niftkMorphologicalSegmentorWidget::setupUi(containerForControlsWidget);

  layout->addWidget(m_ContainerForSelectorWidget, 0, 0);
  layout->addWidget(m_ContainerForToolWidget, 1, 0);
  layout->addWidget(containerForControlsWidget, 2, 0);

  layout->setRowStretch(0, 0);
  layout->setRowStretch(1, 1);
  layout->setRowStretch(2, 0);

  m_TabWidget->setCurrentIndex(0);

  m_ToolSelectorWidget->SetDisplayedToolGroups("Paintbrush");

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

  m_ThresholdingAxialCutOffSlider->layout()->setSpacing(2);
  m_ThresholdingAxialCutOffSlider->setSpinBoxAlignment(Qt::AlignRight);
  m_ThresholdingAxialCutOffSlider->setSingleStep(1.0);
  m_ThresholdingAxialCutOffSlider->setPageStep(2.0);
  m_ThresholdingAxialCutOffSlider->setDecimals(0);
  // Trick alert!
  // So that the width of the spinbox is equal to the other spinboxes:
  m_ThresholdingAxialCutOffSlider->setMaximum(100.0);

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

  this->SetEnabled(false);

  this->connect(m_ThresholdingLowerThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnThresholdLowerValueChanged()));
  this->connect(m_ThresholdingUpperThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnThresholdUpperValueChanged()));
  this->connect(m_ThresholdingAxialCutOffSlider, SIGNAL(valueChanged(double)), SLOT(OnAxialCutOffSliderChanged()));
  this->connect(m_BackButton, SIGNAL(clicked()), SLOT(OnBackButtonClicked()));
  this->connect(m_NextButton, SIGNAL(clicked()), SLOT(OnNextButtonClicked()));
  this->connect(m_ErosionsUpperThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnErosionsUpperThresholdChanged()));
  this->connect(m_ErosionsIterationsSlider, SIGNAL(valueChanged(double)), SLOT(OnErosionsIterationsChanged()));
  this->connect(m_DilationsLowerThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnDilationsLowerThresholdChanged()));
  this->connect(m_DilationsUpperThresholdSlider, SIGNAL(valueChanged(double)), SLOT(OnDilationsUpperThresholdChanged()));
  this->connect(m_DilationsIterationsSlider, SIGNAL(valueChanged(double)), SLOT(OnDilationsIterationsChanged()));
  this->connect(m_RethresholdingBoxSizeSlider, SIGNAL(valueChanged(double)), SLOT(OnRethresholdingSliderChanged()));
  this->connect(m_RestartButton, SIGNAL(clicked()), SLOT(OnRestartButtonClicked()));
  this->connect(m_CancelButton, SIGNAL(clicked()), SLOT(OnCancelButtonClicked()));
}


//-----------------------------------------------------------------------------
MorphologicalSegmentorGUI::~MorphologicalSegmentorGUI()
{
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::SetEnabled(bool enabled)
{
  int tabIndex = enabled ? this->GetTabIndex() : -1;

  for (int i = 0; i < 4; ++i)
  {
    m_TabWidget->setTabEnabled(i, i == tabIndex);
  }

  m_BackButton->setEnabled(tabIndex > 0);
  m_NextButton->setText(tabIndex < 3 ? "Next >" : "Finish");
  m_NextButton->setEnabled(tabIndex >= 0);
  m_CancelButton->setEnabled(tabIndex >= 0);
  m_RestartButton->setEnabled(tabIndex > 0);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::EnableSegmentationWidgets(bool enabled)
{
  int tabIndex = this->GetTabIndex();

  this->SetToolSelectorEnabled(enabled && (tabIndex == 1 || tabIndex == 2));

  this->SetEnabled(enabled);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::SetControlsByReferenceImage(double lowestValue, double highestValue, int numberOfAxialSlices, int upDirection)
{
  bool wasBlocked = this->blockSignals(true);

  double stepSize = 1;
  double pageSize = 10;

  if (std::fabs(highestValue - lowestValue) < 50)
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

  m_ThresholdingAxialCutOffSlider->setMinimum(0);
  m_ThresholdingAxialCutOffSlider->setMaximum(numberOfAxialSlices - 1);
  if (upDirection > 0)
  {
    m_ThresholdingAxialCutOffSlider->setInvertedAppearance(false);
    m_ThresholdingAxialCutOffSlider->setInvertedControls(false);
    m_ThresholdingAxialCutOffSlider->setValue(0);
  }
  else
  {
    m_ThresholdingAxialCutOffSlider->setInvertedAppearance(true);
    m_ThresholdingAxialCutOffSlider->setInvertedControls(true);
    m_ThresholdingAxialCutOffSlider->setValue(numberOfAxialSlices - 1);
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
void MorphologicalSegmentorGUI::SetControlsFromSegmentationNode(mitk::DataNode* segmentationNode)
{
  bool wasBlocked = this->blockSignals(true);

  if (segmentationNode)
  {
    int intValue;
    float floatValue;

    segmentationNode->GetIntProperty("midas.morph.stage", intValue);
    m_TabWidget->setCurrentIndex(intValue);

    segmentationNode->GetFloatProperty("midas.morph.thresholding.lower", floatValue);
    m_ThresholdingLowerThresholdSlider->setValue(floatValue);

    segmentationNode->GetFloatProperty("midas.morph.thresholding.upper", floatValue);
    m_ThresholdingUpperThresholdSlider->setValue(floatValue);

    segmentationNode->GetIntProperty("midas.morph.thresholding.slice", intValue);
    m_ThresholdingAxialCutOffSlider->setValue(intValue);

    segmentationNode->GetFloatProperty("midas.morph.erosion.threshold", floatValue);
    m_ErosionsUpperThresholdSlider->setValue(floatValue);

    segmentationNode->GetIntProperty("midas.morph.erosion.iterations", intValue);
    m_ErosionsIterationsSlider->setValue(intValue);

    segmentationNode->GetFloatProperty("midas.morph.dilation.lower", floatValue);
    m_DilationsLowerThresholdSlider->setValue(floatValue);

    segmentationNode->GetFloatProperty("midas.morph.dilation.upper", floatValue);
    m_DilationsUpperThresholdSlider->setValue(floatValue);

    segmentationNode->GetIntProperty("midas.morph.dilation.iterations", intValue);
    m_DilationsIterationsSlider->setValue(intValue);

    segmentationNode->GetIntProperty("midas.morph.rethresholding.box", intValue);
    m_RethresholdingBoxSizeSlider->setValue(intValue);

    bool finished = true;
    segmentationNode->GetBoolProperty("midas.morph.finished", finished);

    this->SetEnabled(!finished);
  }
  else
  {
    m_TabWidget->setCurrentIndex(0);
    m_ThresholdingLowerThresholdSlider->setValue(0.0);
    m_ThresholdingUpperThresholdSlider->setValue(0.0);
    m_ThresholdingAxialCutOffSlider->setValue(0);
    m_ErosionsUpperThresholdSlider->setValue(0.0);
    m_ErosionsIterationsSlider->setValue(0);
    m_DilationsLowerThresholdSlider->setValue(0.0);
    m_DilationsUpperThresholdSlider->setValue(0.0);
    m_DilationsIterationsSlider->setValue(0);
    m_RethresholdingBoxSizeSlider->setValue(0);
    this->SetEnabled(false);
  }

  this->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int MorphologicalSegmentorGUI::GetTabIndex() const
{
  return m_TabWidget->currentIndex();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::SetTabIndex(int tabIndex)
{
  bool wasBlocked = m_TabWidget->blockSignals(true);
  m_TabWidget->setCurrentIndex(tabIndex);
  m_TabWidget->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::EmitThresholdingValues()
{
  emit ThresholdingValuesChanged(
         m_ThresholdingLowerThresholdSlider->value(),
         m_ThresholdingUpperThresholdSlider->value(),
         static_cast<int>(m_ThresholdingAxialCutOffSlider->value())
       );
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::EmitErosionValues()
{
  emit ErosionsValuesChanged(
         m_ErosionsUpperThresholdSlider->value(),
         static_cast<int>(m_ErosionsIterationsSlider->value())
       );
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::EmitDilationValues()
{
  emit DilationsValuesChanged(
         m_DilationsLowerThresholdSlider->value(),
         m_DilationsUpperThresholdSlider->value(),
         static_cast<int>(m_DilationsIterationsSlider->value())
       );
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::EmitRethresholdingValues()
{
  emit RethresholdingValuesChanged(
         static_cast<int>(m_RethresholdingBoxSizeSlider->value())
      );
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::OnThresholdLowerValueChanged()
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
void MorphologicalSegmentorGUI::OnThresholdUpperValueChanged()
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
void MorphologicalSegmentorGUI::OnAxialCutOffSliderChanged()
{
  this->EmitThresholdingValues();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::OnBackButtonClicked()
{
  int tabIndex = this->GetTabIndex();

  --tabIndex;

  m_TabWidget->setCurrentIndex(tabIndex);
  this->SetEnabled(true);

  emit TabChanged(tabIndex);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::OnNextButtonClicked()
{
  int tabIndex = this->GetTabIndex();

  if (tabIndex < 3)
  {
    ++tabIndex;

    if (tabIndex == 1)
    {
      bool wasBlocked = m_ErosionsUpperThresholdSlider->blockSignals(true);

      m_ErosionsUpperThresholdSlider->setMinimum(m_ThresholdingLowerThresholdSlider->value());
      m_ErosionsUpperThresholdSlider->setMaximum(m_ThresholdingUpperThresholdSlider->value());
      m_ErosionsUpperThresholdSlider->setValue(m_ThresholdingUpperThresholdSlider->value());

      m_ErosionsUpperThresholdSlider->blockSignals(wasBlocked);
    }

    m_TabWidget->setCurrentIndex(tabIndex);
    this->SetEnabled(true);

    emit TabChanged(tabIndex);
  }
  else
  {
    emit this->OKButtonClicked();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::OnErosionsIterationsChanged()
{
  this->EmitErosionValues();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::OnErosionsUpperThresholdChanged()
{
  this->EmitErosionValues();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::OnDilationsLowerThresholdChanged()
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
void MorphologicalSegmentorGUI::OnDilationsUpperThresholdChanged()
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
void MorphologicalSegmentorGUI::OnDilationsIterationsChanged()
{
  this->EmitDilationValues();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::OnRethresholdingSliderChanged()
{
  this->EmitRethresholdingValues();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::OnRestartButtonClicked()
{
  m_TabWidget->setCurrentIndex(0);
  this->SetEnabled(true);

  emit RestartButtonClicked();
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorGUI::OnCancelButtonClicked()
{
  bool wasBlocked;

  wasBlocked = m_TabWidget->blockSignals(true);
  m_TabWidget->setCurrentIndex(0);
  m_TabWidget->blockSignals(wasBlocked);

  wasBlocked = m_ThresholdingLowerThresholdSlider->blockSignals(true);
  m_ThresholdingLowerThresholdSlider->setMinimum(0.0);
  m_ThresholdingLowerThresholdSlider->setMaximum(0.0);
  m_ThresholdingLowerThresholdSlider->setValue(0);
  m_ThresholdingLowerThresholdSlider->blockSignals(wasBlocked);

  wasBlocked = m_ThresholdingUpperThresholdSlider->blockSignals(true);
  m_ThresholdingUpperThresholdSlider->setMinimum(0.0);
  m_ThresholdingUpperThresholdSlider->setMaximum(0.0);
  m_ThresholdingUpperThresholdSlider->setValue(0);
  m_ThresholdingAxialCutOffSlider->blockSignals(wasBlocked);

  wasBlocked = m_ThresholdingAxialCutOffSlider->blockSignals(true);
  m_ThresholdingAxialCutOffSlider->setMinimum(0.0);
  m_ThresholdingAxialCutOffSlider->setMaximum(100.0);
  m_ThresholdingAxialCutOffSlider->setInvertedAppearance(false);
  m_ThresholdingAxialCutOffSlider->setInvertedControls(false);
  m_ThresholdingAxialCutOffSlider->setValue(0);
  m_ThresholdingUpperThresholdSlider->blockSignals(wasBlocked);

  this->SetEnabled(false);

  emit CancelButtonClicked();
}

}
