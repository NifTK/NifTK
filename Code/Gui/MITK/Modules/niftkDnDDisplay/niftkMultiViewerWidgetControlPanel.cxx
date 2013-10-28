/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMultiViewerWidgetControlPanel.h"

#include <mitkLogMacros.h>

//-----------------------------------------------------------------------------
niftkMultiViewerWidgetControlPanel::niftkMultiViewerWidgetControlPanel(QWidget *parent)
: QWidget(parent)
, m_ShowMagnificationControls(true)
, m_ShowShowOptions(true)
, m_ShowWindowLayoutControls(true)
, m_ShowViewNumberControls(true)
, m_ShowDropTypeControls(true)
{
  this->setupUi(this);

  // Default all widgets off except viewer number widgets, until something dropped.
  this->SetSingleViewControlsEnabled(false);

  // This should disable the view binding and drop type controls.
  this->SetViewNumber(1, 1);

  connect(m_SlidersWidget, SIGNAL(SliceIndexChanged(int)), this, SIGNAL(SliceIndexChanged(int)));
  connect(m_SlidersWidget, SIGNAL(TimeStepChanged(int)), this, SIGNAL(TimeStepChanged(int)));
  connect(m_SlidersWidget, SIGNAL(MagnificationChanged(double)), this, SIGNAL(MagnificationChanged(double)));

  connect(m_ShowCursorCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(ShowCursorChanged(bool)));
  connect(m_ShowDirectionAnnotationsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(ShowDirectionAnnotationsChanged(bool)));
  connect(m_Show3DWindowCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(Show3DWindowChanged(bool)));

  connect(m_LayoutWidget, SIGNAL(LayoutChanged(WindowLayout)), this, SLOT(OnLayoutChanged(WindowLayout)));
  connect(m_BindWindowCursorsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(WindowCursorBindingChanged(bool)));
  connect(m_BindWindowMagnificationsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(WindowMagnificationBindingChanged(bool)));

  connect(m_1x1ViewsButton, SIGNAL(clicked()), this, SLOT(On1x1ViewsButtonClicked()));
  connect(m_1x2ViewsButton, SIGNAL(clicked()), this, SLOT(On1x2ViewsButtonClicked()));
  connect(m_1x3ViewsButton, SIGNAL(clicked()), this, SLOT(On1x3ViewsButtonClicked()));
  connect(m_2x1ViewsButton, SIGNAL(clicked()), this, SLOT(On2x1ViewsButtonClicked()));
  connect(m_2x2ViewsButton, SIGNAL(clicked()), this, SLOT(On2x2ViewsButtonClicked()));
  connect(m_2x3ViewsButton, SIGNAL(clicked()), this, SLOT(On2x3ViewsButtonClicked()));
  connect(m_ViewRowsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnViewRowsSpinBoxValueChanged(int)));
  connect(m_ViewColumnsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnViewColumnsSpinBoxValueChanged(int)));

  connect(m_BindViewPositionsCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnViewPositionBindingChanged(bool)));
  connect(m_BindViewCursorsCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnViewCursorBindingChanged(bool)));
  connect(m_BindViewMagnificationsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(ViewMagnificationBindingChanged(bool)));
  connect(m_BindViewLayoutsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(ViewLayoutBindingChanged(bool)));
  connect(m_BindViewGeometriesCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(ViewGeometryBindingChanged(bool)));

  connect(m_DropSingleRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropSingleRadioButtonToggled(bool)));
  connect(m_DropMultipleRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropMultipleRadioButtonToggled(bool)));
  connect(m_DropThumbnailRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropThumbnailRadioButtonToggled(bool)));
  connect(m_DropAccumulateCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(DropAccumulateChanged(bool)));
}


//-----------------------------------------------------------------------------
niftkMultiViewerWidgetControlPanel::~niftkMultiViewerWidgetControlPanel()
{
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreMagnificationControlsVisible() const
{
  return m_ShowMagnificationControls;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetMagnificationControlsVisible(bool visible)
{
  m_ShowMagnificationControls = visible;
  m_SlidersWidget->SetMagnificationControlsVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreShowOptionsVisible() const
{
  return m_ShowShowOptions;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetShowOptionsVisible(bool visible)
{
  m_ShowShowOptions = visible;
  m_ShowOptionsWidget->setVisible(visible);
  m_ShowOptionsSeparator->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreWindowLayoutControlsVisible() const
{
  return m_ShowWindowLayoutControls;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetWindowLayoutControlsVisible(bool visible)
{
  m_ShowWindowLayoutControls = visible;
  m_WindowLayoutSeparator->setVisible(visible);
  m_WindowLayoutWidget->setVisible(visible);
  m_WindowBindingOptionsSeparator->setVisible(visible);
  m_WindowBindingWidget->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreViewNumberControlsVisible() const
{
  return m_ViewNumberWidget->isVisible();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetViewNumberControlsVisible(bool visible)
{
  m_ViewNumberWidget->setVisible(visible);
  m_ViewBindingSeparator->setVisible(visible);
  m_ViewBindingWidget->setVisible(visible);
  m_MultiViewControlsSeparator->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreDropTypeControlsVisible() const
{
  return m_DropTypeWidget->isVisible();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetDropTypeControlsVisible(bool visible)
{
  m_DropTypeWidget->setVisible(visible);
  m_DropTypeSeparator->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreSingleViewControlsEnabled() const
{
  return m_SlidersWidget->isEnabled();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetSingleViewControlsEnabled(bool enabled)
{
  m_SlidersWidget->setEnabled(enabled);
  m_ShowOptionsSeparator->setEnabled(enabled);
  m_ShowOptionsWidget->setEnabled(enabled);
  m_WindowBindingOptionsSeparator->setEnabled(enabled);
  m_WindowBindingWidget->setEnabled(enabled);
  m_WindowLayoutSeparator->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreMultiViewControlsEnabled() const
{
  return m_ViewNumberWidget->isEnabled();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetMultiViewControlsEnabled(bool enabled)
{
  m_ViewNumberWidget->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetSliceIndexTracking(bool tracking)
{
  m_SlidersWidget->SetSliceIndexTracking(tracking);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetTimeStepTracking(bool tracking)
{
  m_SlidersWidget->SetTimeStepTracking(tracking);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetMagnificationTracking(bool tracking)
{
  m_SlidersWidget->SetMagnificationTracking(tracking);
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidgetControlPanel::GetMaxSliceIndex() const
{
  return m_SlidersWidget->GetMaxSliceIndex();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetMaxSliceIndex(int maxSliceIndex)
{
  m_SlidersWidget->SetMaxSliceIndex(maxSliceIndex);
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidgetControlPanel::GetSliceIndex() const
{
  return m_SlidersWidget->GetSliceIndex();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetSliceIndex(int sliceIndex)
{
  m_SlidersWidget->SetSliceIndex(sliceIndex);
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidgetControlPanel::GetMaxTimeStep() const
{
  return m_SlidersWidget->GetMaxTimeStep();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetMaxTimeStep(int maxTimeStep)
{
  m_SlidersWidget->SetMaxTimeStep(maxTimeStep);
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidgetControlPanel::GetTimeStep() const
{
  return m_SlidersWidget->GetTimeStep();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetTimeStep(int timeStep)
{
  m_SlidersWidget->SetTimeStep(timeStep);
}


//-----------------------------------------------------------------------------
double niftkMultiViewerWidgetControlPanel::GetMinMagnification() const
{
  return m_SlidersWidget->GetMinMagnification();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetMinMagnification(double minMagnification)
{
  m_SlidersWidget->SetMinMagnification(minMagnification);
}


//-----------------------------------------------------------------------------
double niftkMultiViewerWidgetControlPanel::GetMaxMagnification() const
{
  return m_SlidersWidget->GetMaxMagnification();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetMaxMagnification(double maxMagnification)
{
  m_SlidersWidget->SetMaxMagnification(maxMagnification);
}


//-----------------------------------------------------------------------------
double niftkMultiViewerWidgetControlPanel::GetMagnification() const
{
  return m_SlidersWidget->GetMagnification();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetMagnification(double magnification)
{
  m_SlidersWidget->SetMagnification(magnification);
}


//-----------------------------------------------------------------------------
WindowLayout niftkMultiViewerWidgetControlPanel::GetLayout() const
{
  return m_LayoutWidget->GetLayout();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetLayout(WindowLayout windowLayout)
{
  bool wasBlocked = m_LayoutWidget->blockSignals(true);
  m_LayoutWidget->SetLayout(windowLayout);
  m_LayoutWidget->blockSignals(wasBlocked);

  m_WindowBindingWidget->setEnabled(::IsMultiWindowLayout(windowLayout));
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreWindowCursorsBound() const
{
  return m_BindWindowCursorsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetWindowCursorsBound(bool bound)
{
  bool wasBlocked = m_BindWindowCursorsCheckBox->blockSignals(true);
  m_BindWindowCursorsCheckBox->setChecked(bound);
  m_BindWindowCursorsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreWindowMagnificationsBound() const
{
  return m_BindWindowMagnificationsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetWindowMagnificationsBound(bool bound)
{
  bool wasBlocked = m_BindWindowMagnificationsCheckBox->blockSignals(true);
  m_BindWindowMagnificationsCheckBox->setChecked(bound);
  m_BindWindowMagnificationsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::IsCursorVisible() const
{
  return m_ShowCursorCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetCursorVisible(bool visible)
{
  bool wasBlocked = m_ShowCursorCheckBox->blockSignals(true);
  m_ShowCursorCheckBox->setChecked(visible);
  m_ShowCursorCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreDirectionAnnotationsVisible() const
{
  return m_ShowDirectionAnnotationsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetDirectionAnnotationsVisible(bool visible)
{
  bool wasBlocked = m_ShowDirectionAnnotationsCheckBox->blockSignals(true);
  m_ShowDirectionAnnotationsCheckBox->setChecked(visible);
  m_ShowDirectionAnnotationsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::Is3DWindowVisible() const
{
  return m_Show3DWindowCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::Set3DWindowVisible(bool visible)
{
  bool wasBlocked = m_Show3DWindowCheckBox->blockSignals(true);
  m_Show3DWindowCheckBox->setChecked(visible);
  m_Show3DWindowCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidgetControlPanel::GetMaxViewRows() const
{
  return m_ViewRowsSpinBox->maximum();
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidgetControlPanel::GetMaxViewColumns() const
{
  return m_ViewColumnsSpinBox->maximum();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetMaxViewNumber(int maxRows, int maxColumns)
{
  bool wasBlocked = m_1x2ViewsButton->blockSignals(true);
  m_1x2ViewsButton->setEnabled(maxColumns >= 2);
  m_1x2ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_1x3ViewsButton->blockSignals(true);
  m_1x3ViewsButton->setEnabled(maxColumns >= 3);
  m_1x3ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_2x1ViewsButton->blockSignals(true);
  m_2x1ViewsButton->setEnabled(maxRows >= 2 && maxColumns >= 1);
  m_2x1ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_2x2ViewsButton->blockSignals(true);
  m_2x2ViewsButton->setEnabled(maxRows >= 2 && maxColumns >= 2);
  m_2x2ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_2x3ViewsButton->blockSignals(true);
  m_2x3ViewsButton->setEnabled(maxRows >= 2 && maxColumns >= 3);
  m_2x3ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_ViewRowsSpinBox->blockSignals(true);
  m_ViewRowsSpinBox->setMaximum(maxRows);
  m_ViewRowsSpinBox->blockSignals(wasBlocked);

  wasBlocked = m_ViewColumnsSpinBox->blockSignals(true);
  m_ViewColumnsSpinBox->setMaximum(maxColumns);
  m_ViewColumnsSpinBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidgetControlPanel::GetViewRows() const
{
  return m_ViewRowsSpinBox->value();
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidgetControlPanel::GetViewColumns() const
{
  return m_ViewColumnsSpinBox->value();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetViewNumber(int rows, int columns)
{
  if (rows > m_ViewRowsSpinBox->maximum() || columns > m_ViewColumnsSpinBox->maximum())
  {
    return;
  }

  bool singleView = rows == 1 && columns == 1;

  bool wasBlocked = m_1x1ViewsButton->blockSignals(true);
  m_1x1ViewsButton->setChecked(singleView);
  m_1x1ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_1x2ViewsButton->blockSignals(true);
  m_1x2ViewsButton->setChecked(rows == 1 && columns == 2);
  m_1x2ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_1x3ViewsButton->blockSignals(true);
  m_1x3ViewsButton->setChecked(rows == 1 && columns == 3);
  m_1x3ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_2x1ViewsButton->blockSignals(true);
  m_2x1ViewsButton->setChecked(rows == 2 && columns == 1);
  m_2x1ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_2x2ViewsButton->blockSignals(true);
  m_2x2ViewsButton->setChecked(rows == 2 && columns == 2);
  m_2x2ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_2x3ViewsButton->blockSignals(true);
  m_2x3ViewsButton->setChecked(rows == 2 && columns == 3);
  m_2x3ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_ViewRowsSpinBox->blockSignals(true);
  m_ViewRowsSpinBox->setValue(rows);
  m_ViewRowsSpinBox->blockSignals(wasBlocked);

  wasBlocked = m_ViewColumnsSpinBox->blockSignals(true);
  m_ViewColumnsSpinBox->setValue(columns);
  m_ViewColumnsSpinBox->blockSignals(wasBlocked);

  m_ViewBindingWidget->setEnabled(!singleView);
  m_DropTypeWidget->setEnabled(!singleView);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreViewPositionsBound() const
{
  return m_BindViewPositionsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetViewPositionsBound(bool bound)
{
  bool wasBlocked = m_BindViewPositionsCheckBox->blockSignals(true);
  m_BindViewPositionsCheckBox->setChecked(bound);
  m_BindViewPositionsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreViewCursorsBound() const
{
  return m_BindViewCursorsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetViewCursorsBound(bool bound)
{
  bool wasBlocked = m_BindViewCursorsCheckBox->blockSignals(true);
  m_BindViewCursorsCheckBox->setChecked(bound);
  m_BindViewCursorsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreViewMagnificationsBound() const
{
  return m_BindViewMagnificationsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetViewMagnificationsBound(bool bound)
{
  bool wasBlocked = m_BindViewMagnificationsCheckBox->blockSignals(true);
  m_BindViewMagnificationsCheckBox->setChecked(bound);
  m_BindViewMagnificationsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreViewLayoutsBound() const
{
  return m_BindViewLayoutsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetViewLayoutsBound(bool bound)
{
  bool wasBlocked = m_BindViewLayoutsCheckBox->blockSignals(true);
  m_BindViewLayoutsCheckBox->setChecked(bound);
  m_BindViewLayoutsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidgetControlPanel::AreViewGeometriesBound() const
{
  return m_BindViewGeometriesCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetViewGeometriesBound(bool bound)
{
  bool wasBlocked = m_BindViewGeometriesCheckBox->blockSignals(true);
  m_BindViewGeometriesCheckBox->setChecked(bound);
  m_BindViewGeometriesCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
MIDASDropType niftkMultiViewerWidgetControlPanel::GetDropType() const
{
  MIDASDropType dropType = MIDAS_DROP_TYPE_SINGLE;

  if (m_DropMultipleRadioButton->isChecked())
  {
    dropType = MIDAS_DROP_TYPE_MULTIPLE;
  }
  else if (m_DropThumbnailRadioButton->isChecked())
  {
    dropType = MIDAS_DROP_TYPE_ALL;
  }

  return dropType;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::SetDropType(MIDASDropType dropType)
{
  switch (dropType)
  {
  case MIDAS_DROP_TYPE_SINGLE:
    m_DropSingleRadioButton->setChecked(true);
    break;
  case MIDAS_DROP_TYPE_MULTIPLE:
    m_DropMultipleRadioButton->setChecked(true);
    break;
  case MIDAS_DROP_TYPE_ALL:
    m_DropThumbnailRadioButton->setChecked(true);
    break;
  default:
    MITK_ERROR << "niftkMultiViewerControlPanel::SetDropType: Invalid MIDASDropType=" << dropType << std::endl;
    break;
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::OnLayoutChanged(WindowLayout windowLayout)
{
  m_WindowBindingWidget->setEnabled(::IsMultiWindowLayout(windowLayout));

  emit LayoutChanged(windowLayout);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::On1x1ViewsButtonClicked()
{
  this->SetViewNumber(1, 1);
  emit ViewNumberChanged(1, 1);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::On1x2ViewsButtonClicked()
{
  this->SetViewNumber(1, 2);
  emit ViewNumberChanged(1, 2);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::On1x3ViewsButtonClicked()
{
  this->SetViewNumber(1, 3);
  emit ViewNumberChanged(1, 3);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::On2x1ViewsButtonClicked()
{
  this->SetViewNumber(2, 1);
  emit ViewNumberChanged(2, 1);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::On2x2ViewsButtonClicked()
{
  this->SetViewNumber(2, 2);
  emit ViewNumberChanged(2, 2);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::On2x3ViewsButtonClicked()
{
  this->SetViewNumber(2, 3);
  emit ViewNumberChanged(2, 3);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::OnViewRowsSpinBoxValueChanged(int rows)
{
  int columns = m_ViewColumnsSpinBox->value();
  this->SetViewNumber(rows, columns);
  emit ViewNumberChanged(rows, columns);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::OnViewColumnsSpinBoxValueChanged(int columns)
{
  int rows = m_ViewRowsSpinBox->value();
  this->SetViewNumber(rows, columns);
  emit ViewNumberChanged(rows, columns);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::OnViewPositionBindingChanged(bool bound)
{
  if (!bound && this->AreViewCursorsBound())
  {
    // Note that this will trigger emitting the ViewCursorBindingChanged(false) signal.
    m_BindViewCursorsCheckBox->setChecked(false);
  }
  emit ViewPositionBindingChanged(bound);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::OnViewCursorBindingChanged(bool bound)
{
  if (bound && !this->AreViewPositionsBound())
  {
    // Note that this will trigger emitting the ViewPositionBindingChanged(true) signal.
    m_BindViewPositionsCheckBox->setChecked(true);
  }
  emit ViewCursorBindingChanged(bound);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::OnDropSingleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(MIDAS_DROP_TYPE_SINGLE);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::OnDropMultipleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(MIDAS_DROP_TYPE_MULTIPLE);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidgetControlPanel::OnDropThumbnailRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(MIDAS_DROP_TYPE_ALL);
  }
}
