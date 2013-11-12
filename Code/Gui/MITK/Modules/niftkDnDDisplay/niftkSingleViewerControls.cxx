/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleViewerControls.h"

#include "ui_niftkSingleViewerControls.h"

#include <ctkDoubleSpinBox.h>

WindowLayout niftkSingleViewerControls::s_MultiWindowLayouts[] = {
  WINDOW_LAYOUT_ORTHO,
  WINDOW_LAYOUT_3H,
  WINDOW_LAYOUT_3V,
  WINDOW_LAYOUT_COR_SAG_H,
  WINDOW_LAYOUT_COR_SAG_V,
  WINDOW_LAYOUT_COR_AX_H,
  WINDOW_LAYOUT_COR_AX_V,
  WINDOW_LAYOUT_SAG_AX_H,
  WINDOW_LAYOUT_SAG_AX_V
};

int const niftkSingleViewerControls::s_MultiWindowLayoutNumber = sizeof(s_MultiWindowLayouts) / sizeof(WindowLayout);

//-----------------------------------------------------------------------------
niftkSingleViewerControls::niftkSingleViewerControls(QWidget *parent)
: QWidget(parent)
, m_ShowShowOptions(true)
, m_ShowWindowLayoutControls(true)
, m_WindowLayout(WINDOW_LAYOUT_UNKNOWN)
{
  ui = new Ui::niftkSingleViewerControls();
  ui->setupUi(parent);

  ui->m_SliceIndexSlider->layout()->setSpacing(3);
  ui->m_SliceIndexSlider->setDecimals(0);
  ui->m_SliceIndexSlider->setTickInterval(1.0);
  ui->m_SliceIndexSlider->setSingleStep(1.0);
  ui->m_SliceIndexSlider->spinBox()->setAlignment(Qt::AlignRight);

  ui->m_TimeStepSlider->layout()->setSpacing(3);
  ui->m_TimeStepSlider->setDecimals(0);
  ui->m_TimeStepSlider->setTickInterval(1.0);
  ui->m_TimeStepSlider->setSingleStep(1.0);
  ui->m_TimeStepSlider->spinBox()->setAlignment(Qt::AlignRight);

  ui->m_MagnificationSlider->layout()->setSpacing(3);
  ui->m_MagnificationSlider->setDecimals(2);
  ui->m_MagnificationSlider->setTickInterval(1.0);
  ui->m_MagnificationSlider->setSingleStep(1.0);
  ui->m_MagnificationSlider->spinBox()->setAlignment(Qt::AlignRight);

  this->connect(ui->m_SliceIndexSlider, SIGNAL(valueChanged(double)), SLOT(OnSliceIndexChanged(double)));
  this->connect(ui->m_TimeStepSlider, SIGNAL(valueChanged(double)), SLOT(OnTimeStepChanged(double)));
  this->connect(ui->m_MagnificationSlider, SIGNAL(valueChanged(double)), SIGNAL(MagnificationChanged(double)));

  this->connect(ui->m_ShowCursorCheckBox, SIGNAL(toggled(bool)), SIGNAL(ShowCursorChanged(bool)));
  this->connect(ui->m_ShowDirectionAnnotationsCheckBox, SIGNAL(toggled(bool)), SIGNAL(ShowDirectionAnnotationsChanged(bool)));
  this->connect(ui->m_Show3DWindowCheckBox, SIGNAL(toggled(bool)), SIGNAL(Show3DWindowChanged(bool)));

  this->connect(ui->m_BindWindowCursorsCheckBox, SIGNAL(toggled(bool)), SIGNAL(WindowCursorBindingChanged(bool)));
  this->connect(ui->m_BindWindowMagnificationsCheckBox, SIGNAL(toggled(bool)), SIGNAL(WindowMagnificationBindingChanged(bool)));

  ui->m_MultiWindowComboBox->addItem("2x2");
  ui->m_MultiWindowComboBox->addItem("3H");
  ui->m_MultiWindowComboBox->addItem("3V");
  ui->m_MultiWindowComboBox->addItem("cor sag H");
  ui->m_MultiWindowComboBox->addItem("cor sag V");
  ui->m_MultiWindowComboBox->addItem("cor ax H");
  ui->m_MultiWindowComboBox->addItem("cor ax V");
  ui->m_MultiWindowComboBox->addItem("sag ax H");
  ui->m_MultiWindowComboBox->addItem("sag ax V");

  ui->m_AxialWindowRadioButton->setChecked(true);

  this->connect(ui->m_AxialWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnAxialWindowRadioButtonToggled(bool)));
  this->connect(ui->m_SagittalWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnSagittalWindowRadioButtonToggled(bool)));
  this->connect(ui->m_CoronalWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnCoronalWindowRadioButtonToggled(bool)));
  this->connect(ui->m_3DWindowRadioButton, SIGNAL(toggled(bool)), SLOT(On3DWindowRadioButtonToggled(bool)));
  this->connect(ui->m_MultiWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnMultiWindowRadioButtonToggled(bool)));
  this->connect(ui->m_MultiWindowComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnMultiWindowComboBoxIndexChanged(int)));
}


//-----------------------------------------------------------------------------
niftkSingleViewerControls::~niftkSingleViewerControls()
{
}

//-----------------------------------------------------------------------------
bool niftkSingleViewerControls::AreMagnificationControlsVisible() const
{
  return ui->m_MagnificationLabel->isVisible() && ui->m_MagnificationSlider->isVisible();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMagnificationControlsVisible(bool visible)
{
  ui->m_MagnificationLabel->setVisible(visible);
  ui->m_MagnificationSlider->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerControls::AreShowOptionsVisible() const
{
  return m_ShowShowOptions;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetShowOptionsVisible(bool visible)
{
  m_ShowShowOptions = visible;
  ui->m_ShowOptionsWidget->setVisible(visible);
  ui->m_ShowOptionsSeparator->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerControls::AreWindowLayoutControlsVisible() const
{
  return m_ShowWindowLayoutControls;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetWindowLayoutControlsVisible(bool visible)
{
  m_ShowWindowLayoutControls = visible;
  ui->m_WindowLayoutSeparator->setVisible(visible);
  ui->m_WindowLayoutWidget->setVisible(visible);
  ui->m_WindowBindingOptionsSeparator->setVisible(visible);
  ui->m_WindowBindingWidget->setVisible(visible);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::OnSliceIndexChanged(double sliceIndex)
{
  emit SliceIndexChanged(static_cast<int>(sliceIndex));
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::OnTimeStepChanged(double timeStep)
{
  emit TimeStepChanged(static_cast<int>(timeStep));
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetSliceIndexTracking(bool tracking)
{
  ui->m_SliceIndexSlider->setTracking(tracking);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetTimeStepTracking(bool tracking)
{
  ui->m_TimeStepSlider->setTracking(tracking);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMagnificationTracking(bool tracking)
{
  ui->m_MagnificationSlider->setTracking(tracking);
}


//-----------------------------------------------------------------------------
int niftkSingleViewerControls::GetMaxSliceIndex() const
{
  return static_cast<int>(ui->m_SliceIndexSlider->maximum());
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMaxSliceIndex(int maxSliceIndex)
{
  bool wasBlocked = ui->m_SliceIndexSlider->blockSignals(true);
  ui->m_SliceIndexSlider->setMaximum(maxSliceIndex);
  ui->m_SliceIndexSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int niftkSingleViewerControls::GetSliceIndex() const
{
  return static_cast<int>(ui->m_SliceIndexSlider->value());
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetSliceIndex(int sliceIndex)
{
  bool wasBlocked = ui->m_SliceIndexSlider->blockSignals(true);
  ui->m_SliceIndexSlider->setValue(sliceIndex);
  ui->m_SliceIndexSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int niftkSingleViewerControls::GetMaxTimeStep() const
{
  return static_cast<int>(ui->m_TimeStepSlider->maximum());
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMaxTimeStep(int maxTimeStep)
{
  bool wasBlocked = ui->m_TimeStepSlider->blockSignals(true);
  ui->m_TimeStepSlider->setMaximum(maxTimeStep);
  ui->m_TimeStepSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int niftkSingleViewerControls::GetTimeStep() const
{
  return static_cast<int>(ui->m_TimeStepSlider->value());
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetTimeStep(int timeStep)
{
  bool wasBlocked = ui->m_TimeStepSlider->blockSignals(true);
  ui->m_TimeStepSlider->setValue(timeStep);
  ui->m_TimeStepSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
double niftkSingleViewerControls::GetMinMagnification() const
{
  return ui->m_MagnificationSlider->minimum();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMinMagnification(double minMagnification)
{
  bool wasBlocked = ui->m_MagnificationSlider->blockSignals(true);
  ui->m_MagnificationSlider->setMinimum(minMagnification);
  ui->m_MagnificationSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
double niftkSingleViewerControls::GetMaxMagnification() const
{
  return ui->m_MagnificationSlider->maximum();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMaxMagnification(double maxMagnification)
{
  bool wasBlocked = ui->m_MagnificationSlider->blockSignals(true);
  ui->m_MagnificationSlider->setMaximum(maxMagnification);
  ui->m_MagnificationSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
double niftkSingleViewerControls::GetMagnification() const
{
  return ui->m_MagnificationSlider->value();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMagnification(double magnification)
{
  bool wasBlocked = ui->m_MagnificationSlider->blockSignals(true);
  ui->m_MagnificationSlider->setValue(magnification);
  ui->m_MagnificationSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
WindowLayout niftkSingleViewerControls::GetWindowLayout() const
{
  return m_WindowLayout;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetWindowLayout(WindowLayout windowLayout)
{
  if (windowLayout == m_WindowLayout)
  {
    // Nothing to do.
    return;
  }

  bool wasBlocked;

  switch (windowLayout)
  {
  case WINDOW_LAYOUT_AXIAL:
    wasBlocked = ui->m_AxialWindowRadioButton->blockSignals(true);
    ui->m_AxialWindowRadioButton->setChecked(true);
    ui->m_AxialWindowRadioButton->blockSignals(wasBlocked);
    break;
  case WINDOW_LAYOUT_SAGITTAL:
    wasBlocked = ui->m_SagittalWindowRadioButton->blockSignals(true);
    ui->m_SagittalWindowRadioButton->setChecked(true);
    ui->m_SagittalWindowRadioButton->blockSignals(wasBlocked);
    break;
  case WINDOW_LAYOUT_CORONAL:
    wasBlocked = ui->m_CoronalWindowRadioButton->blockSignals(true);
    ui->m_CoronalWindowRadioButton->setChecked(true);
    ui->m_CoronalWindowRadioButton->blockSignals(wasBlocked);
    break;
  case WINDOW_LAYOUT_3D:
    wasBlocked = ui->m_3DWindowRadioButton->blockSignals(true);
    ui->m_3DWindowRadioButton->setChecked(true);
    ui->m_3DWindowRadioButton->blockSignals(wasBlocked);
    break;
  default:
    int windowLayoutIndex = 0;
    while (windowLayoutIndex < s_MultiWindowLayoutNumber && windowLayout != s_MultiWindowLayouts[windowLayoutIndex])
    {
      ++windowLayoutIndex;
    }
    if (windowLayoutIndex == s_MultiWindowLayoutNumber)
    {
      // Should not happen.
      return;
    }

    wasBlocked = ui->m_MultiWindowRadioButton->blockSignals(true);
    ui->m_MultiWindowRadioButton->setChecked(true);
    ui->m_MultiWindowRadioButton->blockSignals(wasBlocked);

    wasBlocked = ui->m_MultiWindowComboBox->blockSignals(true);
    ui->m_MultiWindowComboBox->setCurrentIndex(windowLayoutIndex);
    ui->m_MultiWindowComboBox->blockSignals(wasBlocked);
    break;
  }

  m_WindowLayout = windowLayout;

  ui->m_WindowBindingWidget->setEnabled(::IsMultiWindowLayout(windowLayout));

  emit WindowLayoutChanged(windowLayout);

  ui->m_WindowBindingWidget->setEnabled(::IsMultiWindowLayout(windowLayout));
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerControls::AreWindowCursorsBound() const
{
  return ui->m_BindWindowCursorsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetWindowCursorsBound(bool bound)
{
  bool wasBlocked = ui->m_BindWindowCursorsCheckBox->blockSignals(true);
  ui->m_BindWindowCursorsCheckBox->setChecked(bound);
  ui->m_BindWindowCursorsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerControls::AreWindowMagnificationsBound() const
{
  return ui->m_BindWindowMagnificationsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetWindowMagnificationsBound(bool bound)
{
  bool wasBlocked = ui->m_BindWindowMagnificationsCheckBox->blockSignals(true);
  ui->m_BindWindowMagnificationsCheckBox->setChecked(bound);
  ui->m_BindWindowMagnificationsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerControls::IsCursorVisible() const
{
  return ui->m_ShowCursorCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetCursorVisible(bool visible)
{
  bool wasBlocked = ui->m_ShowCursorCheckBox->blockSignals(true);
  ui->m_ShowCursorCheckBox->setChecked(visible);
  ui->m_ShowCursorCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerControls::AreDirectionAnnotationsVisible() const
{
  return ui->m_ShowDirectionAnnotationsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetDirectionAnnotationsVisible(bool visible)
{
  bool wasBlocked = ui->m_ShowDirectionAnnotationsCheckBox->blockSignals(true);
  ui->m_ShowDirectionAnnotationsCheckBox->setChecked(visible);
  ui->m_ShowDirectionAnnotationsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkSingleViewerControls::Is3DWindowVisible() const
{
  return ui->m_Show3DWindowCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::Set3DWindowVisible(bool visible)
{
  bool wasBlocked = ui->m_Show3DWindowCheckBox->blockSignals(true);
  ui->m_Show3DWindowCheckBox->setChecked(visible);
  ui->m_Show3DWindowCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::OnAxialWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  }
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  }
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  }
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::On3DWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_3D);
  }
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::OnMultiWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(s_MultiWindowLayouts[ui->m_MultiWindowComboBox->currentIndex()]);
  }
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::OnMultiWindowComboBoxIndexChanged(int index)
{
  ui->m_MultiWindowRadioButton->setChecked(true);
  this->SetWindowLayout(s_MultiWindowLayouts[index]);
}
