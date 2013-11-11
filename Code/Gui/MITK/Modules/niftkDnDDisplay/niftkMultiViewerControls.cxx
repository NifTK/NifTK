/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMultiViewerControls.h"

#include <mitkLogMacros.h>

//-----------------------------------------------------------------------------
niftkMultiViewerControls::niftkMultiViewerControls(QWidget *parent)
: niftkSingleViewerControls(new QWidget())
, m_ShowViewNumberControls(true)
, m_ShowDropTypeControls(true)
{
  QWidget* singleViewerControlsWidget = qobject_cast<QWidget*>(this->parent());
  this->setParent(parent);

  Ui_niftkMultiViewerControls::setupUi(parent);

  QHBoxLayout* singleViewerControlsLayout = new QHBoxLayout();
  singleViewerControlsLayout->setMargin(0);
  singleViewerControlsLayout->addWidget(singleViewerControlsWidget);
  m_SingleViewerControls->setLayout(singleViewerControlsLayout);

  // Default all widgets off except viewer number widgets, until something dropped.
  this->SetSingleViewControlsEnabled(false);

  // This should disable the view binding and drop type controls.
  this->SetViewNumber(1, 1);

  connect(m_1x1ViewerButton, SIGNAL(clicked()), this, SLOT(On1x1ViewsButtonClicked()));
  connect(m_1x2ViewersButton, SIGNAL(clicked()), this, SLOT(On1x2ViewsButtonClicked()));
  connect(m_1x3ViewersButton, SIGNAL(clicked()), this, SLOT(On1x3ViewsButtonClicked()));
  connect(m_2x1ViewersButton, SIGNAL(clicked()), this, SLOT(On2x1ViewsButtonClicked()));
  connect(m_2x2ViewersButton, SIGNAL(clicked()), this, SLOT(On2x2ViewsButtonClicked()));
  connect(m_2x3ViewersButton, SIGNAL(clicked()), this, SLOT(On2x3ViewsButtonClicked()));
  connect(m_ViewerRowsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnViewRowsSpinBoxValueChanged(int)));
  connect(m_ViewerColumnsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnViewColumnsSpinBoxValueChanged(int)));

  connect(m_BindViewerPositionsCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnViewPositionBindingChanged(bool)));
  connect(m_BindViewerCursorsCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnViewCursorBindingChanged(bool)));
  connect(m_BindViewerMagnificationsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(ViewMagnificationBindingChanged(bool)));
  connect(m_BindViewerLayoutsCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(ViewLayoutBindingChanged(bool)));
  connect(m_BindViewerGeometriesCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(ViewGeometryBindingChanged(bool)));

  connect(m_DropSingleRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropSingleRadioButtonToggled(bool)));
  connect(m_DropMultipleRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropMultipleRadioButtonToggled(bool)));
  connect(m_DropThumbnailRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropThumbnailRadioButtonToggled(bool)));
  connect(m_DropAccumulateCheckBox, SIGNAL(toggled(bool)), this, SIGNAL(DropAccumulateChanged(bool)));
}


//-----------------------------------------------------------------------------
niftkMultiViewerControls::~niftkMultiViewerControls()
{
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerControls::AreViewNumberControlsVisible() const
{
  return m_ViewerNumberWidget->isVisible();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetViewNumberControlsVisible(bool visible)
{
  m_ViewerNumberWidget->setVisible(visible);
  m_ViewerBindingSeparator->setVisible(visible);
  m_ViewerBindingWidget->setVisible(visible);
  m_MultiViewerControlsSeparator->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerControls::AreDropTypeControlsVisible() const
{
  return m_DropTypeWidget->isVisible();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetDropTypeControlsVisible(bool visible)
{
  m_DropTypeWidget->setVisible(visible);
  m_DropTypeSeparator->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerControls::AreSingleViewControlsEnabled() const
{
//  return m_SlidersWidget->isEnabled();
  return m_SingleViewerControls->isEnabled();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetSingleViewControlsEnabled(bool enabled)
{
  m_SingleViewerControls->setEnabled(enabled);
//  m_SlidersWidget->setEnabled(enabled);
//  m_ShowOptionsSeparator->setEnabled(enabled);
//  m_ShowOptionsWidget->setEnabled(enabled);
//  m_WindowBindingOptionsSeparator->setEnabled(enabled);
//  m_WindowBindingWidget->setEnabled(enabled);
//  m_WindowLayoutSeparator->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerControls::AreMultiViewControlsEnabled() const
{
  return m_ViewerNumberWidget->isEnabled();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetMultiViewControlsEnabled(bool enabled)
{
  m_ViewerNumberWidget->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
int niftkMultiViewerControls::GetMaxViewRows() const
{
  return m_ViewerRowsSpinBox->maximum();
}


//-----------------------------------------------------------------------------
int niftkMultiViewerControls::GetMaxViewColumns() const
{
  return m_ViewerColumnsSpinBox->maximum();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetMaxViewNumber(int maxRows, int maxColumns)
{
  bool wasBlocked = m_1x2ViewersButton->blockSignals(true);
  m_1x2ViewersButton->setEnabled(maxColumns >= 2);
  m_1x2ViewersButton->blockSignals(wasBlocked);

  wasBlocked = m_1x3ViewersButton->blockSignals(true);
  m_1x3ViewersButton->setEnabled(maxColumns >= 3);
  m_1x3ViewersButton->blockSignals(wasBlocked);

  wasBlocked = m_2x1ViewersButton->blockSignals(true);
  m_2x1ViewersButton->setEnabled(maxRows >= 2 && maxColumns >= 1);
  m_2x1ViewersButton->blockSignals(wasBlocked);

  wasBlocked = m_2x2ViewersButton->blockSignals(true);
  m_2x2ViewersButton->setEnabled(maxRows >= 2 && maxColumns >= 2);
  m_2x2ViewersButton->blockSignals(wasBlocked);

  wasBlocked = m_2x3ViewersButton->blockSignals(true);
  m_2x3ViewersButton->setEnabled(maxRows >= 2 && maxColumns >= 3);
  m_2x3ViewersButton->blockSignals(wasBlocked);

  wasBlocked = m_ViewerRowsSpinBox->blockSignals(true);
  m_ViewerRowsSpinBox->setMaximum(maxRows);
  m_ViewerRowsSpinBox->blockSignals(wasBlocked);

  wasBlocked = m_ViewerColumnsSpinBox->blockSignals(true);
  m_ViewerColumnsSpinBox->setMaximum(maxColumns);
  m_ViewerColumnsSpinBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int niftkMultiViewerControls::GetViewRows() const
{
  return m_ViewerRowsSpinBox->value();
}


//-----------------------------------------------------------------------------
int niftkMultiViewerControls::GetViewColumns() const
{
  return m_ViewerColumnsSpinBox->value();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetViewNumber(int rows, int columns)
{
  if (rows > m_ViewerRowsSpinBox->maximum() || columns > m_ViewerColumnsSpinBox->maximum())
  {
    return;
  }

  bool singleView = rows == 1 && columns == 1;

  bool wasBlocked = m_1x1ViewerButton->blockSignals(true);
  m_1x1ViewerButton->setChecked(singleView);
  m_1x1ViewerButton->blockSignals(wasBlocked);

  wasBlocked = m_1x2ViewersButton->blockSignals(true);
  m_1x2ViewersButton->setChecked(rows == 1 && columns == 2);
  m_1x2ViewersButton->blockSignals(wasBlocked);

  wasBlocked = m_1x3ViewersButton->blockSignals(true);
  m_1x3ViewersButton->setChecked(rows == 1 && columns == 3);
  m_1x3ViewersButton->blockSignals(wasBlocked);

  wasBlocked = m_2x1ViewersButton->blockSignals(true);
  m_2x1ViewersButton->setChecked(rows == 2 && columns == 1);
  m_2x1ViewersButton->blockSignals(wasBlocked);

  wasBlocked = m_2x2ViewersButton->blockSignals(true);
  m_2x2ViewersButton->setChecked(rows == 2 && columns == 2);
  m_2x2ViewersButton->blockSignals(wasBlocked);

  wasBlocked = m_2x3ViewersButton->blockSignals(true);
  m_2x3ViewersButton->setChecked(rows == 2 && columns == 3);
  m_2x3ViewersButton->blockSignals(wasBlocked);

  wasBlocked = m_ViewerRowsSpinBox->blockSignals(true);
  m_ViewerRowsSpinBox->setValue(rows);
  m_ViewerRowsSpinBox->blockSignals(wasBlocked);

  wasBlocked = m_ViewerColumnsSpinBox->blockSignals(true);
  m_ViewerColumnsSpinBox->setValue(columns);
  m_ViewerColumnsSpinBox->blockSignals(wasBlocked);

  m_ViewerBindingWidget->setEnabled(!singleView);
  m_DropTypeWidget->setEnabled(!singleView);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerControls::AreViewPositionsBound() const
{
  return m_BindViewerPositionsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetViewPositionsBound(bool bound)
{
  bool wasBlocked = m_BindViewerPositionsCheckBox->blockSignals(true);
  m_BindViewerPositionsCheckBox->setChecked(bound);
  m_BindViewerPositionsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerControls::AreViewCursorsBound() const
{
  return m_BindViewerCursorsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetViewCursorsBound(bool bound)
{
  bool wasBlocked = m_BindViewerCursorsCheckBox->blockSignals(true);
  m_BindViewerCursorsCheckBox->setChecked(bound);
  m_BindViewerCursorsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerControls::AreViewMagnificationsBound() const
{
  return m_BindViewerMagnificationsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetViewMagnificationsBound(bool bound)
{
  bool wasBlocked = m_BindViewerMagnificationsCheckBox->blockSignals(true);
  m_BindViewerMagnificationsCheckBox->setChecked(bound);
  m_BindViewerMagnificationsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerControls::AreViewLayoutsBound() const
{
  return m_BindViewerLayoutsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetViewLayoutsBound(bool bound)
{
  bool wasBlocked = m_BindViewerLayoutsCheckBox->blockSignals(true);
  m_BindViewerLayoutsCheckBox->setChecked(bound);
  m_BindViewerLayoutsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerControls::AreViewGeometriesBound() const
{
  return m_BindViewerGeometriesCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetViewGeometriesBound(bool bound)
{
  bool wasBlocked = m_BindViewerGeometriesCheckBox->blockSignals(true);
  m_BindViewerGeometriesCheckBox->setChecked(bound);
  m_BindViewerGeometriesCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
DnDDisplayDropType niftkMultiViewerControls::GetDropType() const
{
  DnDDisplayDropType dropType = DNDDISPLAY_DROP_SINGLE;

  if (m_DropMultipleRadioButton->isChecked())
  {
    dropType = DNDDISPLAY_DROP_MULTIPLE;
  }
  else if (m_DropThumbnailRadioButton->isChecked())
  {
    dropType = DNDDISPLAY_DROP_ALL;
  }

  return dropType;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::SetDropType(DnDDisplayDropType dropType)
{
  switch (dropType)
  {
  case DNDDISPLAY_DROP_SINGLE:
    m_DropSingleRadioButton->setChecked(true);
    break;
  case DNDDISPLAY_DROP_MULTIPLE:
    m_DropMultipleRadioButton->setChecked(true);
    break;
  case DNDDISPLAY_DROP_ALL:
    m_DropThumbnailRadioButton->setChecked(true);
    break;
  default:
    MITK_ERROR << "niftkMultiViewerControlPanel::SetDropType: Invalid DnDDisplayDropType=" << dropType << std::endl;
    break;
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::On1x1ViewsButtonClicked()
{
  this->SetViewNumber(1, 1);
  emit ViewNumberChanged(1, 1);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::On1x2ViewsButtonClicked()
{
  this->SetViewNumber(1, 2);
  emit ViewNumberChanged(1, 2);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::On1x3ViewsButtonClicked()
{
  this->SetViewNumber(1, 3);
  emit ViewNumberChanged(1, 3);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::On2x1ViewsButtonClicked()
{
  this->SetViewNumber(2, 1);
  emit ViewNumberChanged(2, 1);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::On2x2ViewsButtonClicked()
{
  this->SetViewNumber(2, 2);
  emit ViewNumberChanged(2, 2);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::On2x3ViewsButtonClicked()
{
  this->SetViewNumber(2, 3);
  emit ViewNumberChanged(2, 3);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::OnViewRowsSpinBoxValueChanged(int rows)
{
  int columns = m_ViewerColumnsSpinBox->value();
  this->SetViewNumber(rows, columns);
  emit ViewNumberChanged(rows, columns);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::OnViewColumnsSpinBoxValueChanged(int columns)
{
  int rows = m_ViewerRowsSpinBox->value();
  this->SetViewNumber(rows, columns);
  emit ViewNumberChanged(rows, columns);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::OnViewPositionBindingChanged(bool bound)
{
  if (!bound && this->AreViewCursorsBound())
  {
    // Note that this will trigger emitting the ViewCursorBindingChanged(false) signal.
    m_BindViewerCursorsCheckBox->setChecked(false);
  }
  emit ViewPositionBindingChanged(bound);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::OnViewCursorBindingChanged(bool bound)
{
  if (bound && !this->AreViewPositionsBound())
  {
    // Note that this will trigger emitting the ViewPositionBindingChanged(true) signal.
    m_BindViewerPositionsCheckBox->setChecked(true);
  }
  emit ViewCursorBindingChanged(bound);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::OnDropSingleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(DNDDISPLAY_DROP_SINGLE);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::OnDropMultipleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(DNDDISPLAY_DROP_MULTIPLE);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerControls::OnDropThumbnailRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(DNDDISPLAY_DROP_ALL);
  }
}
