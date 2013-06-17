/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASBindWidget.h"

#include <mitkLogMacros.h>

//-----------------------------------------------------------------------------
QmitkMIDASBindWidget::QmitkMIDASBindWidget(QWidget *parent)
: QWidget(parent)
, m_BindType(MIDAS_BIND_NONE)
{
  this->setupUi(this);

  m_BindLayoutsCheckBox->setChecked(false);
  m_BindCursorsCheckBox->setChecked(false);
  m_BindMagnificationsCheckBox->setChecked(false);
  m_BindGeometriesCheckBox->setChecked(false);

  connect(m_BindLayoutsCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnBindLayoutsCheckBoxToggled(bool)));
  connect(m_BindCursorsCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnBindCursorsCheckBoxToggled(bool)));
  connect(m_BindMagnificationsCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnBindMagnificationsCheckBoxToggled(bool)));
  connect(m_BindGeometriesCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnBindGeometriesCheckBoxToggled(bool)));
}


//-----------------------------------------------------------------------------
QmitkMIDASBindWidget::~QmitkMIDASBindWidget()
{
}


//-----------------------------------------------------------------------------
bool QmitkMIDASBindWidget::AreLayoutsBound() const
{
  return m_BindType & MIDAS_BIND_LAYOUT;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASBindWidget::AreCursorsBound() const
{
  return m_BindType & MIDAS_BIND_CURSORS;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASBindWidget::AreMagnificationsBound() const
{
  return m_BindType & MIDAS_BIND_MAGNIFICATION;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASBindWidget::AreGeometriesBound() const
{
  return m_BindType & MIDAS_BIND_GEOMETRY;
}


//-----------------------------------------------------------------------------
void QmitkMIDASBindWidget::SetBindType(int bindType)
{
  if (bindType == m_BindType)
  {
    // Nothing to do.
    return;
  }

  bool wasBlocked = m_BindLayoutsCheckBox->blockSignals(true);
  m_BindLayoutsCheckBox->setChecked(bindType & MIDAS_BIND_LAYOUT);
  m_BindLayoutsCheckBox->blockSignals(wasBlocked);

  wasBlocked = m_BindCursorsCheckBox->blockSignals(true);
  m_BindCursorsCheckBox->setChecked(bindType & MIDAS_BIND_CURSORS);
  m_BindCursorsCheckBox->blockSignals(wasBlocked);

  wasBlocked = m_BindMagnificationsCheckBox->blockSignals(true);
  m_BindMagnificationsCheckBox->setChecked(bindType & MIDAS_BIND_MAGNIFICATION);
  m_BindMagnificationsCheckBox->blockSignals(wasBlocked);

  wasBlocked = m_BindGeometriesCheckBox->blockSignals(true);
  m_BindGeometriesCheckBox->setChecked(bindType & MIDAS_BIND_GEOMETRY);
  m_BindGeometriesCheckBox->blockSignals(wasBlocked);

  m_BindType = bindType;
}


//-----------------------------------------------------------------------------
void QmitkMIDASBindWidget::OnBindLayoutsCheckBoxToggled(bool checked)
{
  if (checked)
  {
    m_BindType |= MIDAS_BIND_LAYOUT;
  }
  else
  {
    m_BindType &= ~MIDAS_BIND_LAYOUT;
  }
  emit BindTypeChanged();
}


//-----------------------------------------------------------------------------
void QmitkMIDASBindWidget::OnBindCursorsCheckBoxToggled(bool checked)
{
  if (checked)
  {
    m_BindType |= MIDAS_BIND_CURSORS;
  }
  else
  {
    m_BindType &= ~MIDAS_BIND_CURSORS;
  }
  emit BindTypeChanged();
}


//-----------------------------------------------------------------------------
void QmitkMIDASBindWidget::OnBindMagnificationsCheckBoxToggled(bool checked)
{
  if (checked)
  {
    m_BindType |= MIDAS_BIND_MAGNIFICATION;
  }
  else
  {
    m_BindType &= ~MIDAS_BIND_MAGNIFICATION;
  }
  emit BindTypeChanged();
}


//-----------------------------------------------------------------------------
void QmitkMIDASBindWidget::OnBindGeometriesCheckBoxToggled(bool checked)
{
  if (checked)
  {
    m_BindType |= MIDAS_BIND_GEOMETRY;
  }
  else
  {
    m_BindType &= ~MIDAS_BIND_GEOMETRY;
  }
  emit BindTypeChanged();
}
