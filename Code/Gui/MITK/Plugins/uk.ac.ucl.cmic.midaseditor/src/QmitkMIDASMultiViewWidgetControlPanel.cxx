#include "QmitkMIDASMultiViewWidgetControlPanel.h"

#include <mitkLogMacros.h>

//-----------------------------------------------------------------------------
QmitkMIDASMultiViewWidgetControlPanel::QmitkMIDASMultiViewWidgetControlPanel(QWidget *parent)
: QWidget(parent)
{
  this->setupUi(this);

  // Default all widgets off except view number widgets, until something dropped.
  this->SetSliceAndMagnificationControlsEnabled(false);
  this->SetLayoutControlsEnabled(false);
  this->SetViewNumberControlsEnabled(true);

  // This should disable the view binding and drop type controls.
  this->SetViewNumber(1, 1);

  m_BindWindowPanningCheckBox->setVisible(false);
  m_BindWindowZoomCheckBox->setVisible(false);

  connect(m_SlidersWidget, SIGNAL(SliceIndexChanged(int)), this, SIGNAL(SliceIndexChanged(int)));
  connect(m_SlidersWidget, SIGNAL(TimeStepChanged(int)), this, SIGNAL(TimeStepChanged(int)));
  connect(m_SlidersWidget, SIGNAL(MagnificationChanged(double)), this, SIGNAL(MagnificationChanged(double)));

  connect(m_LayoutWidget, SIGNAL(LayoutChanged(MIDASLayout)), this, SIGNAL(LayoutChanged(MIDASLayout)));
  connect(m_ShowCursorsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(CursorVisibilityChanged(bool)));

  connect(m_1x1ViewButton, SIGNAL(clicked()), this, SLOT(On1x1ViewButtonClicked()));
  connect(m_1x2ViewsButton, SIGNAL(clicked()), this, SLOT(On1x2ViewsButtonClicked()));
  connect(m_1x3ViewsButton, SIGNAL(clicked()), this, SLOT(On1x3ViewsButtonClicked()));
  connect(m_2x2ViewsButton, SIGNAL(clicked()), this, SLOT(On2x2ViewsButtonClicked()));
  connect(m_ViewRowsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnViewRowsSpinBoxValueChanged(int)));
  connect(m_ViewColumnsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnViewColumnsSpinBoxValueChanged(int)));

  connect(m_ViewBindingWidget, SIGNAL(BindTypeChanged()), this, SIGNAL(ViewBindingTypeChanged()));

  connect(m_DropSingleRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropSingleRadioButtonToggled(bool)));
  connect(m_DropMultipleRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropMultipleRadioButtonToggled(bool)));
  connect(m_DropThumbnailRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropThumbnailRadioButtonToggled(bool)));
  connect(m_DropAccumulateCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(DropAccumulateChanged(bool)));
}


//-----------------------------------------------------------------------------
QmitkMIDASMultiViewWidgetControlPanel::~QmitkMIDASMultiViewWidgetControlPanel()
{
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetSliceAndMagnificationControlsEnabled(bool enabled)
{
  m_SlidersWidget->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetLayoutControlsEnabled(bool enabled)
{
  m_LayoutGroupBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetViewNumberControlsEnabled(bool enabled)
{
  m_ViewNumberGroupBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetViewBindingControlsEnabled(bool enabled)
{
  m_ViewBindingGroupBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetDropTypeControlsEnabled(bool enabled)
{
  m_DropTypeGroupBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetMagnificationControlsVisible(bool visible)
{
  m_SlidersWidget->SetMagnificationControlsVisible(visible);
  m_SlidersGroupBox->setTitle(visible ? "Slice && magnification" : "Slice");
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetViewNumberControlsVisible(bool visible)
{
  m_ViewNumberGroupBox->setVisible(visible);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetDropTypeControlsVisible(bool visible)
{
  m_DropTypeGroupBox->setVisible(visible);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetSliceIndexTracking(bool tracking)
{
  m_SlidersWidget->SetSliceIndexTracking(tracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetTimeStepTracking(bool tracking)
{
  m_SlidersWidget->SetTimeStepTracking(tracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetMagnificationTracking(bool tracking)
{
  m_SlidersWidget->SetMagnificationTracking(tracking);
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidgetControlPanel::GetMaxSliceIndex() const
{
  return m_SlidersWidget->GetMaxSliceIndex();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetMaxSliceIndex(int maxSliceIndex)
{
  m_SlidersWidget->SetMaxSliceIndex(maxSliceIndex);
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidgetControlPanel::GetSliceIndex() const
{
  return m_SlidersWidget->GetSliceIndex();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetSliceIndex(int sliceIndex)
{
  m_SlidersWidget->SetSliceIndex(sliceIndex);
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidgetControlPanel::GetMaxTimeStep() const
{
  return m_SlidersWidget->GetMaxTimeStep();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetMaxTimeStep(int maxTimeStep)
{
  m_SlidersWidget->SetMaxTimeStep(maxTimeStep);
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidgetControlPanel::GetTimeStep() const
{
  return m_SlidersWidget->GetTimeStep();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetTimeStep(int timeStep)
{
  m_SlidersWidget->SetTimeStep(timeStep);
}


//-----------------------------------------------------------------------------
double QmitkMIDASMultiViewWidgetControlPanel::GetMinMagnification() const
{
  return m_SlidersWidget->GetMinMagnification();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetMinMagnification(double minMagnification)
{
  m_SlidersWidget->SetMinMagnification(minMagnification);
}


//-----------------------------------------------------------------------------
double QmitkMIDASMultiViewWidgetControlPanel::GetMaxMagnification() const
{
  return m_SlidersWidget->GetMaxMagnification();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetMaxMagnification(double maxMagnification)
{
  m_SlidersWidget->SetMaxMagnification(maxMagnification);
}


//-----------------------------------------------------------------------------
double QmitkMIDASMultiViewWidgetControlPanel::GetMagnification() const
{
  return m_SlidersWidget->GetMagnification();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetMagnification(double magnification)
{
  m_SlidersWidget->SetMagnification(magnification);
}


//-----------------------------------------------------------------------------
MIDASLayout QmitkMIDASMultiViewWidgetControlPanel::GetLayout() const
{
  return m_LayoutWidget->GetLayout();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetLayout(MIDASLayout layout)
{
  bool wasBlocked = m_LayoutWidget->blockSignals(true);
  m_LayoutWidget->SetLayout(layout);
  m_LayoutWidget->blockSignals(wasBlocked);
}

//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidgetControlPanel::IsCursorVisible() const
{
  return m_ShowCursorsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetCursorVisible(bool visible)
{
  bool wasBlocked = m_ShowCursorsCheckBox->blockSignals(true);
  m_ShowCursorsCheckBox->setChecked(visible);
  m_ShowCursorsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidgetControlPanel::GetMaxViewRows() const
{
  return m_ViewRowsSpinBox->maximum();
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidgetControlPanel::GetMaxViewColumns() const
{
  return m_ViewColumnsSpinBox->maximum();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetMaxViewNumber(int maxRows, int maxColumns)
{
  bool wasBlocked = m_1x2ViewsButton->blockSignals(true);
  m_1x2ViewsButton->setEnabled(maxColumns >= 2);
  m_1x2ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_1x3ViewsButton->blockSignals(true);
  m_1x3ViewsButton->setEnabled(maxColumns >= 3);
  m_1x3ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_2x2ViewsButton->blockSignals(true);
  m_2x2ViewsButton->setEnabled(maxRows >= 2 && maxColumns >= 2);
  m_2x2ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_ViewRowsSpinBox->blockSignals(true);
  m_ViewRowsSpinBox->setMaximum(maxRows);
  m_ViewRowsSpinBox->blockSignals(wasBlocked);

  wasBlocked = m_ViewColumnsSpinBox->blockSignals(true);
  m_ViewColumnsSpinBox->setMaximum(maxColumns);
  m_ViewColumnsSpinBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidgetControlPanel::GetViewRows() const
{
  return m_ViewRowsSpinBox->value();
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidgetControlPanel::GetViewColumns() const
{
  return m_ViewColumnsSpinBox->value();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::SetViewNumber(int rows, int columns)
{
  bool singleView = rows == 1 && columns == 1;

  bool wasBlocked = m_1x1ViewButton->blockSignals(true);
  m_1x1ViewButton->setChecked(singleView);
  m_1x1ViewButton->blockSignals(wasBlocked);

  wasBlocked = m_1x2ViewsButton->blockSignals(true);
  m_1x2ViewsButton->setChecked(rows == 1 && columns == 2);
  m_1x2ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_1x3ViewsButton->blockSignals(true);
  m_1x3ViewsButton->setChecked(rows == 1 && columns == 3);
  m_1x3ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_2x2ViewsButton->blockSignals(true);
  m_2x2ViewsButton->setChecked(rows == 2 && columns == 2);
  m_2x2ViewsButton->blockSignals(wasBlocked);

  wasBlocked = m_ViewRowsSpinBox->blockSignals(true);
  m_ViewRowsSpinBox->setValue(rows);
  m_ViewRowsSpinBox->blockSignals(wasBlocked);

  wasBlocked = m_ViewColumnsSpinBox->blockSignals(true);
  m_ViewColumnsSpinBox->setValue(columns);
  m_ViewColumnsSpinBox->blockSignals(wasBlocked);

  this->SetViewBindingControlsEnabled(!singleView);
  this->SetDropTypeControlsEnabled(!singleView);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidgetControlPanel::AreViewLayoutsBound() const
{
  return m_ViewBindingWidget->AreLayoutsBound();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidgetControlPanel::AreViewCursorsBound() const
{
  return m_ViewBindingWidget->AreCursorsBound();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidgetControlPanel::AreViewMagnificationsBound() const
{
  return m_ViewBindingWidget->AreMagnificationsBound();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidgetControlPanel::AreViewGeometriesBound() const
{
  return m_ViewBindingWidget->AreGeometriesBound();
}


//-----------------------------------------------------------------------------
MIDASDropType QmitkMIDASMultiViewWidgetControlPanel::GetDropType() const
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
void QmitkMIDASMultiViewWidgetControlPanel::SetDropType(MIDASDropType dropType)
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
    MITK_ERROR << "QmitkMIDASMultiViewControlPanel::SetDropType: Invalid MIDASDropType=" << dropType << std::endl;
    break;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::On1x1ViewButtonClicked()
{
  this->SetViewNumber(1, 1);
  emit ViewNumberChanged(1, 1);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::On1x2ViewsButtonClicked()
{
  this->SetViewNumber(1, 2);
  emit ViewNumberChanged(1, 2);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::On1x3ViewsButtonClicked()
{
  this->SetViewNumber(1, 3);
  emit ViewNumberChanged(1, 3);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::On2x2ViewsButtonClicked()
{
  this->SetViewNumber(2, 2);
  emit ViewNumberChanged(2, 2);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::OnViewRowsSpinBoxValueChanged(int rows)
{
  int columns = m_ViewColumnsSpinBox->value();
  this->SetViewNumber(rows, columns);
  emit ViewNumberChanged(rows, columns);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::OnViewColumnsSpinBoxValueChanged(int columns)
{
  int rows = m_ViewRowsSpinBox->value();
  this->SetViewNumber(rows, columns);
  emit ViewNumberChanged(rows, columns);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::OnDropSingleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(MIDAS_DROP_TYPE_SINGLE);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::OnDropMultipleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(MIDAS_DROP_TYPE_MULTIPLE);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidgetControlPanel::OnDropThumbnailRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(MIDAS_DROP_TYPE_ALL);
  }
}
