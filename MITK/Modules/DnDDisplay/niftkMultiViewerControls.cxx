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


namespace niftk
{

//-----------------------------------------------------------------------------
MultiViewerControls::MultiViewerControls(QWidget *parent)
: SingleViewerControls(new QWidget())
, m_ShowViewerNumberControls(true)
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
  this->SetSingleViewerControlsEnabled(false);

  // This should disable the viewer binding and drop type controls.
  this->SetViewerNumber(1, 1);

  this->connect(m_1x1ViewerButton, SIGNAL(clicked()), SLOT(On1x1ViewerButtonClicked()));
  this->connect(m_1x2ViewersButton, SIGNAL(clicked()), SLOT(On1x2ViewersButtonClicked()));
  this->connect(m_1x3ViewersButton, SIGNAL(clicked()), SLOT(On1x3ViewersButtonClicked()));
  this->connect(m_2x1ViewersButton, SIGNAL(clicked()), SLOT(On2x1ViewersButtonClicked()));
  this->connect(m_2x2ViewersButton, SIGNAL(clicked()), SLOT(On2x2ViewersButtonClicked()));
  this->connect(m_2x3ViewersButton, SIGNAL(clicked()), SLOT(On2x3ViewersButtonClicked()));
  this->connect(m_ViewerRowsSpinBox, SIGNAL(valueChanged(int)), SLOT(OnViewerRowsSpinBoxValueChanged(int)));
  this->connect(m_ViewerColumnsSpinBox, SIGNAL(valueChanged(int)), SLOT(OnViewerColumnsSpinBoxValueChanged(int)));

  this->connect(m_BindViewerPositionsCheckBox, SIGNAL(toggled(bool)), SLOT(OnViewerPositionBindingChanged(bool)));
  this->connect(m_BindViewerCursorsCheckBox, SIGNAL(toggled(bool)), SLOT(OnViewerCursorBindingChanged(bool)));
  this->connect(m_BindViewerMagnificationsCheckBox, SIGNAL(toggled(bool)), SIGNAL(ViewerMagnificationBindingChanged(bool)));
  this->connect(m_BindViewerWindowLayoutsCheckBox, SIGNAL(toggled(bool)), SLOT(OnViewerWindowLayoutBindingChanged(bool)));
  this->connect(m_BindViewerGeometriesCheckBox, SIGNAL(toggled(bool)), SIGNAL(ViewerGeometryBindingChanged(bool)));

  this->connect(m_DropSingleRadioButton, SIGNAL(toggled(bool)), SLOT(OnDropSingleRadioButtonToggled(bool)));
  this->connect(m_DropMultipleRadioButton, SIGNAL(toggled(bool)), SLOT(OnDropMultipleRadioButtonToggled(bool)));
  this->connect(m_DropThumbnailRadioButton, SIGNAL(toggled(bool)), SLOT(OnDropThumbnailRadioButtonToggled(bool)));
  this->connect(m_DropAccumulateCheckBox, SIGNAL(toggled(bool)), SIGNAL(DropAccumulateChanged(bool)));
}


//-----------------------------------------------------------------------------
MultiViewerControls::~MultiViewerControls()
{
}


//-----------------------------------------------------------------------------
bool MultiViewerControls::AreViewerNumberControlsVisible() const
{
  return m_ViewerNumberWidget->isVisible();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetViewerNumberControlsVisible(bool visible)
{
  m_ViewerNumberWidget->setVisible(visible);
  m_ViewerBindingSeparator->setVisible(visible);
  m_ViewerBindingWidget->setVisible(visible);
  m_MultiViewerControlsSeparator->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool MultiViewerControls::AreDropTypeControlsVisible() const
{
  return m_DropTypeWidget->isVisible();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetDropTypeControlsVisible(bool visible)
{
  m_DropTypeWidget->setVisible(visible);
  m_DropTypeSeparator->setVisible(visible);
}


//-----------------------------------------------------------------------------
bool MultiViewerControls::AreSingleViewerControlsEnabled() const
{
//  return m_SlidersWidget->isEnabled();
  return m_SingleViewerControls->isEnabled();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetSingleViewerControlsEnabled(bool enabled)
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
bool MultiViewerControls::AreMultiViewerControlsEnabled() const
{
  return m_ViewerNumberWidget->isEnabled();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetMultiViewerControlsEnabled(bool enabled)
{
  m_ViewerNumberWidget->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
int MultiViewerControls::GetMaxViewerRows() const
{
  return m_ViewerRowsSpinBox->maximum();
}


//-----------------------------------------------------------------------------
int MultiViewerControls::GetMaxViewerColumns() const
{
  return m_ViewerColumnsSpinBox->maximum();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetMaxViewerNumber(int maxRows, int maxColumns)
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
int MultiViewerControls::GetViewerRows() const
{
  return m_ViewerRowsSpinBox->value();
}


//-----------------------------------------------------------------------------
int MultiViewerControls::GetViewerColumns() const
{
  return m_ViewerColumnsSpinBox->value();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetViewerNumber(int rows, int columns)
{
  if (rows > m_ViewerRowsSpinBox->maximum() || columns > m_ViewerColumnsSpinBox->maximum())
  {
    return;
  }

  bool singleViewer = rows == 1 && columns == 1;

  bool wasBlocked = m_1x1ViewerButton->blockSignals(true);
  m_1x1ViewerButton->setChecked(singleViewer);
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

  m_ViewerBindingWidget->setEnabled(!singleViewer);
}


//-----------------------------------------------------------------------------
bool MultiViewerControls::AreViewerPositionsBound() const
{
  return m_BindViewerPositionsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetViewerPositionsBound(bool bound)
{
  bool wasBlocked = m_BindViewerPositionsCheckBox->blockSignals(true);
  m_BindViewerPositionsCheckBox->setChecked(bound);
  m_BindViewerPositionsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool MultiViewerControls::AreViewerCursorsBound() const
{
  return m_BindViewerCursorsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetViewerCursorsBound(bool bound)
{
  bool wasBlocked = m_BindViewerCursorsCheckBox->blockSignals(true);
  m_BindViewerCursorsCheckBox->setChecked(bound);
  m_BindViewerCursorsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool MultiViewerControls::AreViewerMagnificationsBound() const
{
  return m_BindViewerMagnificationsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetViewerMagnificationsBound(bool bound)
{
  bool wasBlocked = m_BindViewerMagnificationsCheckBox->blockSignals(true);
  m_BindViewerMagnificationsCheckBox->setChecked(bound);
  m_BindViewerMagnificationsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool MultiViewerControls::AreViewerWindowLayoutsBound() const
{
  return m_BindViewerWindowLayoutsCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetViewerWindowLayoutsBound(bool bound)
{
  bool wasBlocked = m_BindViewerWindowLayoutsCheckBox->blockSignals(true);
  m_BindViewerWindowLayoutsCheckBox->setChecked(bound);
  m_BindViewerWindowLayoutsCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
bool MultiViewerControls::AreViewerGeometriesBound() const
{
  return m_BindViewerGeometriesCheckBox->isChecked();
}


//-----------------------------------------------------------------------------
void MultiViewerControls::SetViewerGeometriesBound(bool bound)
{
  bool wasBlocked = m_BindViewerGeometriesCheckBox->blockSignals(true);
  m_BindViewerGeometriesCheckBox->setChecked(bound);
  m_BindViewerGeometriesCheckBox->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
DnDDisplayDropType MultiViewerControls::GetDropType() const
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
void MultiViewerControls::SetDropType(DnDDisplayDropType dropType)
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
    MITK_ERROR << "MultiViewerControlPanel::SetDropType: Invalid DnDDisplayDropType=" << dropType << std::endl;
    break;
  }
}


//-----------------------------------------------------------------------------
void MultiViewerControls::On1x1ViewerButtonClicked()
{
  this->SetViewerNumber(1, 1);
  emit ViewerNumberChanged(1, 1);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::On1x2ViewersButtonClicked()
{
  this->SetViewerNumber(1, 2);
  emit ViewerNumberChanged(1, 2);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::On1x3ViewersButtonClicked()
{
  this->SetViewerNumber(1, 3);
  emit ViewerNumberChanged(1, 3);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::On2x1ViewersButtonClicked()
{
  this->SetViewerNumber(2, 1);
  emit ViewerNumberChanged(2, 1);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::On2x2ViewersButtonClicked()
{
  this->SetViewerNumber(2, 2);
  emit ViewerNumberChanged(2, 2);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::On2x3ViewersButtonClicked()
{
  this->SetViewerNumber(2, 3);
  emit ViewerNumberChanged(2, 3);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::OnViewerRowsSpinBoxValueChanged(int rows)
{
  int columns = m_ViewerColumnsSpinBox->value();
  this->SetViewerNumber(rows, columns);
  emit ViewerNumberChanged(rows, columns);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::OnViewerColumnsSpinBoxValueChanged(int columns)
{
  int rows = m_ViewerRowsSpinBox->value();
  this->SetViewerNumber(rows, columns);
  emit ViewerNumberChanged(rows, columns);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::OnViewerPositionBindingChanged(bool bound)
{
  if (!bound && this->AreViewerCursorsBound())
  {
    // Note that this will trigger emitting the ViewCursorBindingChanged(false) signal.
    m_BindViewerCursorsCheckBox->setChecked(false);
  }
  emit ViewerPositionBindingChanged(bound);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::OnViewerCursorBindingChanged(bool bound)
{
  if (bound && !this->AreViewerWindowLayoutsBound())
  {
    // Note that this will trigger emitting the ViewerWindowLayoutBindingChanged(true) signal.
    m_BindViewerWindowLayoutsCheckBox->setChecked(true);
  }
  if (bound && !this->AreViewerPositionsBound())
  {
    // Note that this will trigger emitting the ViewerPositionBindingChanged(true) signal.
    m_BindViewerPositionsCheckBox->setChecked(true);
  }
  emit ViewerCursorBindingChanged(bound);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::OnViewerWindowLayoutBindingChanged(bool bound)
{
  if (!bound && this->AreViewerCursorsBound())
  {
    // Note that this will trigger emitting the ViewCursorBindingChanged(false) signal.
    m_BindViewerCursorsCheckBox->setChecked(false);
  }
  emit ViewerWindowLayoutBindingChanged(bound);
}


//-----------------------------------------------------------------------------
void MultiViewerControls::OnDropSingleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(DNDDISPLAY_DROP_SINGLE);
  }
}


//-----------------------------------------------------------------------------
void MultiViewerControls::OnDropMultipleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(DNDDISPLAY_DROP_MULTIPLE);
  }
}


//-----------------------------------------------------------------------------
void MultiViewerControls::OnDropThumbnailRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    emit DropTypeChanged(DNDDISPLAY_DROP_ALL);
  }
}

}
