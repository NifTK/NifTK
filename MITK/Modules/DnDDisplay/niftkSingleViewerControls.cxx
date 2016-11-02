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

#include <assert.h>

#include <QInputDialog>

#include <ctkDoubleSpinBox.h>

namespace niftk
{

WindowLayout SingleViewerControls::s_MultiWindowLayouts[] = {
  WINDOW_LAYOUT_ORTHO,
  WINDOW_LAYOUT_ORTHO_NO_3D,
  WINDOW_LAYOUT_3H,
  WINDOW_LAYOUT_3V,
  WINDOW_LAYOUT_COR_SAG_H,
  WINDOW_LAYOUT_COR_SAG_V,
  WINDOW_LAYOUT_COR_AX_H,
  WINDOW_LAYOUT_COR_AX_V,
  WINDOW_LAYOUT_SAG_AX_H,
  WINDOW_LAYOUT_SAG_AX_V
};

int const SingleViewerControls::s_MultiWindowLayoutNumber = sizeof(s_MultiWindowLayouts) / sizeof(WindowLayout);

//-----------------------------------------------------------------------------
SingleViewerControls::SingleViewerControls(QWidget *parent)
: QWidget(parent)
, m_ShowShowOptions(true)
, m_ShowWindowLayoutControls(true)
, m_WindowLayout(WINDOW_LAYOUT_UNKNOWN)
{
  ui = new Ui::niftkSingleViewerControls();
  ui->setupUi(parent);

  // Disable synchronising the decimal precision. Synchronise width only.
  ui->m_SliceSlider->setSynchronizeSiblings(ctkSliderWidget::SynchronizeWidth);
  ui->m_TimeStepSlider->setSynchronizeSiblings(ctkSliderWidget::SynchronizeWidth);
  ui->m_MagnificationSlider->setSynchronizeSiblings(ctkSliderWidget::SynchronizeWidth);

  ui->m_SliceSlider->layout()->setSpacing(3);
  ui->m_SliceSlider->setDecimals(0);
  ui->m_SliceSlider->setTickInterval(1.0);
  ui->m_SliceSlider->setSingleStep(1.0);
  ui->m_SliceSlider->spinBox()->setAlignment(Qt::AlignRight);

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

  this->connect(ui->m_SliceSlider, SIGNAL(valueChanged(double)), SLOT(OnSelectedSliceChanged(double)));
  this->connect(ui->m_TimeStepSlider, SIGNAL(valueChanged(double)), SLOT(OnTimeStepChanged(double)));
  this->connect(ui->m_MagnificationSlider, SIGNAL(valueChanged(double)), SIGNAL(MagnificationChanged(double)));

  this->connect(ui->m_ShowCursorCheckBox, SIGNAL(toggled(bool)), SIGNAL(ShowCursorChanged(bool)));
  this->connect(ui->m_ShowDirectionAnnotationsCheckBox, SIGNAL(toggled(bool)), SIGNAL(ShowDirectionAnnotationsChanged(bool)));
  this->connect(ui->m_ShowPositionAnnotationCheckBox, SIGNAL(toggled(bool)), SIGNAL(ShowPositionAnnotationChanged(bool)));
  this->connect(ui->m_ShowIntensityAnnotationCheckBox, SIGNAL(toggled(bool)), SIGNAL(ShowIntensityAnnotationChanged(bool)));
  this->connect(ui->m_ShowPropertyAnnotationCheckBox, SIGNAL(toggled(bool)), SIGNAL(ShowPropertyAnnotationChanged(bool)));
  this->connect(ui->m_PropertiesForAnnotationLabel, SIGNAL(clicked()), SLOT(OnPropertiesForAnnotationLabelClicked()));

  this->connect(ui->m_BindWindowCursorsCheckBox, SIGNAL(toggled(bool)), SIGNAL(WindowCursorBindingChanged(bool)));
  this->connect(ui->m_BindWindowMagnificationsCheckBox, SIGNAL(toggled(bool)), SIGNAL(WindowMagnificationBindingChanged(bool)));

  ui->m_MultiWindowComboBox->addItem("2x2");
  ui->m_MultiWindowComboBox->addItem("2x2 no 3D");
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
SingleViewerControls::~SingleViewerControls()
{
}

//-----------------------------------------------------------------------------
bool SingleViewerControls::AreMagnificationControlsVisible() const
{
  return ui->m_MagnificationLabel->isVisible() && ui->m_MagnificationSlider->isVisible();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetMagnificationControlsVisible(bool visible)
{
  ui->m_MagnificationLabel->setVisible(visible);
  ui->m_MagnificationSlider->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool SingleViewerControls::AreMagnificationControlsEnabled() const
{
  return ui->m_MagnificationLabel->isEnabled() && ui->m_MagnificationSlider->isEnabled();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetMagnificationControlsEnabled(bool enabled)
{
  ui->m_MagnificationLabel->setEnabled(enabled);
  ui->m_MagnificationSlider->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
bool SingleViewerControls::AreShowOptionsVisible() const
{
  return m_ShowShowOptions;
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetShowOptionsVisible(bool visible)
{
  m_ShowShowOptions = visible;
  ui->m_ShowOptionsWidget->setVisible(visible);
  ui->m_ShowOptionsSeparator->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool SingleViewerControls::AreWindowLayoutControlsVisible() const
{
  return m_ShowWindowLayoutControls;
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetWindowLayoutControlsVisible(bool visible)
{
  m_ShowWindowLayoutControls = visible;
  ui->m_WindowLayoutSeparator->setVisible(visible);
  ui->m_WindowLayoutWidget->setVisible(visible);
  ui->m_WindowBindingOptionsSeparator->setVisible(visible);
  ui->m_WindowBindingWidget->setVisible(visible);
}


//-----------------------------------------------------------------------------
void SingleViewerControls::OnSelectedSliceChanged(double selectedSlice)
{
  emit SelectedSliceChanged(static_cast<int>(selectedSlice));
}


//-----------------------------------------------------------------------------
void SingleViewerControls::OnTimeStepChanged(double timeStep)
{
  emit TimeStepChanged(static_cast<int>(timeStep));
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetSliceTracking(bool tracking)
{
  ui->m_SliceSlider->setTracking(tracking);
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetTimeStepTracking(bool tracking)
{
  ui->m_TimeStepSlider->setTracking(tracking);
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetMagnificationTracking(bool tracking)
{
  ui->m_MagnificationSlider->setTracking(tracking);
}


//-----------------------------------------------------------------------------
int SingleViewerControls::GetMaxSlice() const
{
  return static_cast<int>(ui->m_SliceSlider->maximum());
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetMaxSlice(int maxSlice)
{
  bool wasBlocked = ui->m_SliceSlider->blockSignals(true);
  ui->m_SliceSlider->setMaximum(maxSlice);
  ui->m_SliceSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int SingleViewerControls::GetSelectedSlice() const
{
  return static_cast<int>(ui->m_SliceSlider->value());
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetSelectedSlice(int selectedSlice)
{
  bool wasBlocked = ui->m_SliceSlider->blockSignals(true);
  ui->m_SliceSlider->setValue(selectedSlice);
  ui->m_SliceSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int SingleViewerControls::GetMaxTimeStep() const
{
  return static_cast<int>(ui->m_TimeStepSlider->maximum());
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetMaxTimeStep(int maxTimeStep)
{
  bool wasBlocked = ui->m_TimeStepSlider->blockSignals(true);
  ui->m_TimeStepSlider->setMaximum(maxTimeStep);
  ui->m_TimeStepSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int SingleViewerControls::GetTimeStep() const
{
  return static_cast<int>(ui->m_TimeStepSlider->value());
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetTimeStep(int timeStep)
{
  bool wasBlocked = ui->m_TimeStepSlider->blockSignals(true);
  ui->m_TimeStepSlider->setValue(timeStep);
  ui->m_TimeStepSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
double SingleViewerControls::GetMinMagnification() const
{
  return ui->m_MagnificationSlider->minimum();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetMinMagnification(double minMagnification)
{
  bool wasBlocked = ui->m_MagnificationSlider->blockSignals(true);
  ui->m_MagnificationSlider->setMinimum(minMagnification);
  ui->m_MagnificationSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
double SingleViewerControls::GetMaxMagnification() const
{
  return ui->m_MagnificationSlider->maximum();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetMaxMagnification(double maxMagnification)
{
  bool wasBlocked = ui->m_MagnificationSlider->blockSignals(true);
  ui->m_MagnificationSlider->setMaximum(maxMagnification);
  ui->m_MagnificationSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
double SingleViewerControls::GetMagnification() const
{
  return ui->m_MagnificationSlider->value();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetMagnification(double magnification)
{
  bool wasBlocked = ui->m_MagnificationSlider->blockSignals(true);
  ui->m_MagnificationSlider->setValue(magnification);
  ui->m_MagnificationSlider->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
WindowLayout SingleViewerControls::GetWindowLayout() const
{
  return m_WindowLayout;
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetWindowLayout(WindowLayout windowLayout)
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
    while (windowLayoutIndex < s_MultiWindowLayoutNumber
           && s_MultiWindowLayouts[windowLayoutIndex] != windowLayout)
    {
      ++windowLayoutIndex;
    }

    if (windowLayoutIndex == s_MultiWindowLayoutNumber)
    {
      /// This can be WINDOW_LAYOUT_UNKNOWN or WINDOW_LAYOUT_AS_ACQUIRED.
      /// We switch off everything.
      wasBlocked = ui->m_AxialWindowRadioButton->blockSignals(true);
      ui->m_AxialWindowRadioButton->setChecked(false);
      ui->m_AxialWindowRadioButton->blockSignals(wasBlocked);
      wasBlocked = ui->m_SagittalWindowRadioButton->blockSignals(true);
      ui->m_SagittalWindowRadioButton->setChecked(false);
      ui->m_SagittalWindowRadioButton->blockSignals(wasBlocked);
      wasBlocked = ui->m_CoronalWindowRadioButton->blockSignals(true);
      ui->m_CoronalWindowRadioButton->setChecked(false);
      ui->m_CoronalWindowRadioButton->blockSignals(wasBlocked);
      wasBlocked = ui->m_3DWindowRadioButton->blockSignals(true);
      ui->m_3DWindowRadioButton->setChecked(false);
      ui->m_3DWindowRadioButton->blockSignals(wasBlocked);
      wasBlocked = ui->m_MultiWindowRadioButton->blockSignals(true);
      ui->m_MultiWindowRadioButton->setChecked(false);
      ui->m_MultiWindowRadioButton->blockSignals(wasBlocked);
      wasBlocked = ui->m_MultiWindowComboBox->blockSignals(true);
      ui->m_MultiWindowComboBox->setCurrentIndex(-1);
      ui->m_MultiWindowComboBox->blockSignals(wasBlocked);
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

  ui->m_WindowBindingWidget->setEnabled(niftk::IsMultiWindowLayout(windowLayout));
}


//-----------------------------------------------------------------------------
bool SingleViewerControls::AreWindowCursorsBound() const
{
  return ui->m_BindWindowCursorsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetWindowCursorsBound(bool bound)
{
  bool wasBlocked = ui->m_BindWindowCursorsCheckBox->blockSignals(true);
  ui->m_BindWindowCursorsCheckBox->setChecked(bound);
  ui->m_BindWindowCursorsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool SingleViewerControls::AreWindowMagnificationsBound() const
{
  return ui->m_BindWindowMagnificationsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetWindowMagnificationsBound(bool bound)
{
  bool wasBlocked = ui->m_BindWindowMagnificationsCheckBox->blockSignals(true);
  ui->m_BindWindowMagnificationsCheckBox->setChecked(bound);
  ui->m_BindWindowMagnificationsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool SingleViewerControls::IsCursorVisible() const
{
  return ui->m_ShowCursorCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetCursorVisible(bool visible)
{
  bool wasBlocked = ui->m_ShowCursorCheckBox->blockSignals(true);
  ui->m_ShowCursorCheckBox->setChecked(visible);
  ui->m_ShowCursorCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool SingleViewerControls::AreDirectionAnnotationsVisible() const
{
  return ui->m_ShowDirectionAnnotationsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetDirectionAnnotationsVisible(bool visible)
{
  bool wasBlocked = ui->m_ShowDirectionAnnotationsCheckBox->blockSignals(true);
  ui->m_ShowDirectionAnnotationsCheckBox->setChecked(visible);
  ui->m_ShowDirectionAnnotationsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool SingleViewerControls::IsPositionAnnotationVisible() const
{
  return ui->m_ShowPositionAnnotationCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetPositionAnnotationVisible(bool visible)
{
  bool wasBlocked = ui->m_ShowPositionAnnotationCheckBox->blockSignals(true);
  ui->m_ShowPositionAnnotationCheckBox->setChecked(visible);
  ui->m_ShowPositionAnnotationCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool SingleViewerControls::IsIntensityAnnotationVisible() const
{
  return ui->m_ShowIntensityAnnotationCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetIntensityAnnotationVisible(bool visible)
{
  bool wasBlocked = ui->m_ShowIntensityAnnotationCheckBox->blockSignals(true);
  ui->m_ShowIntensityAnnotationCheckBox->setChecked(visible);
  ui->m_ShowIntensityAnnotationCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool SingleViewerControls::IsPropertyAnnotationVisible() const
{
  return ui->m_ShowPropertyAnnotationCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void SingleViewerControls::SetPropertyAnnotationVisible(bool visible)
{
  bool wasBlocked = ui->m_ShowPropertyAnnotationCheckBox->blockSignals(true);
  ui->m_ShowPropertyAnnotationCheckBox->setChecked(visible);
  ui->m_ShowPropertyAnnotationCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
QStringList SingleViewerControls::GetPropertiesForAnnotation() const
{
  return m_PropertiesForAnnotation;
}

//-----------------------------------------------------------------------------
void SingleViewerControls::SetPropertiesForAnnotation(const QStringList& propertiesForAnnotation)
{
  m_PropertiesForAnnotation = propertiesForAnnotation;
}


//-----------------------------------------------------------------------------
void SingleViewerControls::OnPropertiesForAnnotationLabelClicked()
{
  bool ok;
  QString propertyNames = QInputDialog::getText(
        this, tr("Property annotations"),
        tr("Please give the comma separated list of property names:"),
        QLineEdit::Normal, m_PropertiesForAnnotation.join(", "), &ok);
  if (ok)
  {
    QStringList properties;
    for (const QString& propertyName: propertyNames.split(","))
    {
      QString property = propertyName.trimmed();
      if (!property.isEmpty())
      {
        properties.push_back(property);
      }
    }

    m_PropertiesForAnnotation = properties;

    bool wasBlocked = ui->m_ShowPropertyAnnotationCheckBox->blockSignals(true);
    ui->m_ShowPropertyAnnotationCheckBox->setChecked(!properties.isEmpty());
    ui->m_ShowPropertyAnnotationCheckBox->blockSignals(wasBlocked);

    emit PropertiesForAnnotationChanged();
  }
}


//-----------------------------------------------------------------------------
void SingleViewerControls::OnAxialWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
    emit WindowLayoutChanged(WINDOW_LAYOUT_AXIAL);
  }
}


//-----------------------------------------------------------------------------
void SingleViewerControls::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
    emit WindowLayoutChanged(WINDOW_LAYOUT_SAGITTAL);
  }
}


//-----------------------------------------------------------------------------
void SingleViewerControls::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
    emit WindowLayoutChanged(WINDOW_LAYOUT_CORONAL);
  }
}


//-----------------------------------------------------------------------------
void SingleViewerControls::On3DWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_3D);
    emit WindowLayoutChanged(WINDOW_LAYOUT_3D);
  }
}


//-----------------------------------------------------------------------------
void SingleViewerControls::OnMultiWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(s_MultiWindowLayouts[ui->m_MultiWindowComboBox->currentIndex()]);
    emit WindowLayoutChanged(s_MultiWindowLayouts[ui->m_MultiWindowComboBox->currentIndex()]);
  }
}


//-----------------------------------------------------------------------------
void SingleViewerControls::OnMultiWindowComboBoxIndexChanged(int index)
{
  ui->m_MultiWindowRadioButton->setChecked(true);
  this->SetWindowLayout(s_MultiWindowLayouts[index]);
  emit WindowLayoutChanged(s_MultiWindowLayouts[index]);
}

}
