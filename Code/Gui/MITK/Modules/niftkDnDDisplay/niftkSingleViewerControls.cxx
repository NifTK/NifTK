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

//-----------------------------------------------------------------------------
niftkSingleViewerControls::niftkSingleViewerControls(QWidget *parent)
: QWidget(parent)
, m_ShowMagnificationControls(true)
, m_ShowShowOptions(true)
, m_ShowWindowLayoutControls(true)
{
  ui = new Ui::niftkSingleViewerControls();
  ui->setupUi(parent);

  connect(ui->m_SlidersWidget, SIGNAL(SliceIndexChanged(int)), this, SIGNAL(SliceIndexChanged(int)));
  connect(ui->m_SlidersWidget, SIGNAL(TimeStepChanged(int)), this, SIGNAL(TimeStepChanged(int)));
  connect(ui->m_SlidersWidget, SIGNAL(MagnificationChanged(double)), this, SIGNAL(MagnificationChanged(double)));

  connect(ui->m_ShowCursorCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(ShowCursorChanged(bool)));
  connect(ui->m_ShowDirectionAnnotationsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(ShowDirectionAnnotationsChanged(bool)));
  connect(ui->m_Show3DWindowCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(Show3DWindowChanged(bool)));

  connect(ui->m_WindowLayoutWidget, SIGNAL(LayoutChanged(WindowLayout)), this, SLOT(OnLayoutChanged(WindowLayout)));
  connect(ui->m_BindWindowCursorsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(WindowCursorBindingChanged(bool)));
  connect(ui->m_BindWindowMagnificationsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(WindowMagnificationBindingChanged(bool)));
}


//-----------------------------------------------------------------------------
niftkSingleViewerControls::~niftkSingleViewerControls()
{
}

//-----------------------------------------------------------------------------
bool niftkSingleViewerControls::AreMagnificationControlsVisible() const
{
  return m_ShowMagnificationControls;
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMagnificationControlsVisible(bool visible)
{
  m_ShowMagnificationControls = visible;
  ui->m_SlidersWidget->SetMagnificationControlsVisible(visible);
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
void niftkSingleViewerControls::SetSliceIndexTracking(bool tracking)
{
  ui->m_SlidersWidget->SetSliceIndexTracking(tracking);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetTimeStepTracking(bool tracking)
{
  ui->m_SlidersWidget->SetTimeStepTracking(tracking);
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMagnificationTracking(bool tracking)
{
  ui->m_SlidersWidget->SetMagnificationTracking(tracking);
}


//-----------------------------------------------------------------------------
int niftkSingleViewerControls::GetMaxSliceIndex() const
{
  return ui->m_SlidersWidget->GetMaxSliceIndex();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMaxSliceIndex(int maxSliceIndex)
{
  ui->m_SlidersWidget->SetMaxSliceIndex(maxSliceIndex);
}


//-----------------------------------------------------------------------------
int niftkSingleViewerControls::GetSliceIndex() const
{
  return ui->m_SlidersWidget->GetSliceIndex();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetSliceIndex(int sliceIndex)
{
  ui->m_SlidersWidget->SetSliceIndex(sliceIndex);
}


//-----------------------------------------------------------------------------
int niftkSingleViewerControls::GetMaxTimeStep() const
{
  return ui->m_SlidersWidget->GetMaxTimeStep();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMaxTimeStep(int maxTimeStep)
{
  ui->m_SlidersWidget->SetMaxTimeStep(maxTimeStep);
}


//-----------------------------------------------------------------------------
int niftkSingleViewerControls::GetTimeStep() const
{
  return ui->m_SlidersWidget->GetTimeStep();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetTimeStep(int timeStep)
{
  ui->m_SlidersWidget->SetTimeStep(timeStep);
}


//-----------------------------------------------------------------------------
double niftkSingleViewerControls::GetMinMagnification() const
{
  return ui->m_SlidersWidget->GetMinMagnification();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMinMagnification(double minMagnification)
{
  ui->m_SlidersWidget->SetMinMagnification(minMagnification);
}


//-----------------------------------------------------------------------------
double niftkSingleViewerControls::GetMaxMagnification() const
{
  return ui->m_SlidersWidget->GetMaxMagnification();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMaxMagnification(double maxMagnification)
{
  ui->m_SlidersWidget->SetMaxMagnification(maxMagnification);
}


//-----------------------------------------------------------------------------
double niftkSingleViewerControls::GetMagnification() const
{
  return ui->m_SlidersWidget->GetMagnification();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetMagnification(double magnification)
{
  ui->m_SlidersWidget->SetMagnification(magnification);
}


//-----------------------------------------------------------------------------
WindowLayout niftkSingleViewerControls::GetLayout() const
{
  return ui->m_WindowLayoutWidget->GetLayout();
}


//-----------------------------------------------------------------------------
void niftkSingleViewerControls::SetLayout(WindowLayout windowLayout)
{
  bool wasBlocked = ui->m_WindowLayoutWidget->blockSignals(true);
  ui->m_WindowLayoutWidget->SetLayout(windowLayout);
  ui->m_WindowLayoutWidget->blockSignals(wasBlocked);

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
void niftkSingleViewerControls::OnLayoutChanged(WindowLayout windowLayout)
{
  ui->m_WindowBindingWidget->setEnabled(::IsMultiWindowLayout(windowLayout));

  emit LayoutChanged(windowLayout);
}
