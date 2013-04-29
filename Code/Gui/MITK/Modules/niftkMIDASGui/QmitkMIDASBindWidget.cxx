/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASBindWidget.h"
#include <QtDebug>

QmitkMIDASBindWidget::QmitkMIDASBindWidget(QWidget *parent)
{
  m_CurrentBindType = MIDAS_BIND_NONE;
  setupUi(this);
}

QmitkMIDASBindWidget::~QmitkMIDASBindWidget()
{

}

void QmitkMIDASBindWidget::setupUi(QWidget* parent)
{
  Ui_QmitkMIDASBindWidget::setupUi(parent);
  m_BindNoneCheckBox->setCheckState(Qt::Checked);
  m_BindCursorsCheckBox->setCheckState(Qt::Unchecked);
  m_BindMagnificationCheckBox->setCheckState(Qt::Unchecked);
  m_BindGeometryCheckBox->setCheckState(Qt::Unchecked);

  connect(m_BindNoneCheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnNoneCheckBoxStateChanged(int)));
  connect(m_BindCursorsCheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnCursorsCheckBoxStateChanged(int)));
  connect(m_BindMagnificationCheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnMagnificationCheckBoxStateChanged(int)));
  connect(m_BindGeometryCheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnGeometryCheckBoxStateChanged(int)));
}

void QmitkMIDASBindWidget::SetBlockSignals(bool block)
{
  m_BindNoneCheckBox->blockSignals(block);
  m_BindCursorsCheckBox->blockSignals(block);
  m_BindMagnificationCheckBox->blockSignals(block);
  m_BindGeometryCheckBox->blockSignals(block);
}

bool QmitkMIDASBindWidget::IsGeometryBound() const
{
  if (m_CurrentBindType == MIDAS_BIND_GEOMETRY_WITHOUT_MAGNIFICATION
      || m_CurrentBindType == MIDAS_BIND_GEOMETRY_WITH_MAGNIFICATION)
  {
    return true;
  }
  else
  {
    return false;
  }
}

bool QmitkMIDASBindWidget::AreCursorsBound() const
{
  if (m_CurrentBindType == MIDAS_BIND_CURSORS
      || m_CurrentBindType == MIDAS_BIND_MAGNIFICATION_WITH_CURSORS
      || m_CurrentBindType == MIDAS_BIND_GEOMETRY_WITHOUT_MAGNIFICATION
      || m_CurrentBindType == MIDAS_BIND_GEOMETRY_WITH_MAGNIFICATION
      )
  {
    return true;
  }
  else
  {
    return false;
  }
}

bool QmitkMIDASBindWidget::IsMagnificationBound() const
{
  if (m_CurrentBindType == MIDAS_BIND_MAGNIFICATION_WITHOUT_CURSORS
      || m_CurrentBindType == MIDAS_BIND_MAGNIFICATION_WITH_CURSORS
      || m_CurrentBindType == MIDAS_BIND_GEOMETRY_WITH_MAGNIFICATION
      )
  {
    return true;
  }
  else
  {
    return false;
  }
}

void QmitkMIDASBindWidget::SetToBindType(MIDASBindType bindType)
{
  if (bindType == m_CurrentBindType)
  {
    // Nothing to do.
    return;
  }

  this->SetBlockSignals(true);

  switch(bindType)
  {
  case MIDAS_BIND_NONE:
    m_BindNoneCheckBox->setCheckState(Qt::Checked);
    m_BindCursorsCheckBox->setCheckState(Qt::Unchecked);
    m_BindMagnificationCheckBox->setCheckState(Qt::Unchecked);
    m_BindGeometryCheckBox->setCheckState(Qt::Unchecked);
    break;
  case MIDAS_BIND_CURSORS:
    m_BindNoneCheckBox->setCheckState(Qt::Unchecked);
    m_BindCursorsCheckBox->setCheckState(Qt::Checked);
    m_BindMagnificationCheckBox->setCheckState(Qt::Unchecked);
    m_BindGeometryCheckBox->setCheckState(Qt::Unchecked);
    break;
  case MIDAS_BIND_MAGNIFICATION_WITHOUT_CURSORS:
    m_BindNoneCheckBox->setCheckState(Qt::Unchecked);
    m_BindCursorsCheckBox->setCheckState(Qt::Unchecked);
    m_BindMagnificationCheckBox->setCheckState(Qt::Checked);
    m_BindGeometryCheckBox->setCheckState(Qt::Unchecked);
    break;
  case MIDAS_BIND_MAGNIFICATION_WITH_CURSORS:
    m_BindNoneCheckBox->setCheckState(Qt::Unchecked);
    m_BindCursorsCheckBox->setCheckState(Qt::Checked);
    m_BindMagnificationCheckBox->setCheckState(Qt::Checked);
    m_BindGeometryCheckBox->setCheckState(Qt::Unchecked);
    break;
  case MIDAS_BIND_GEOMETRY_WITH_MAGNIFICATION:
    m_BindNoneCheckBox->setCheckState(Qt::Unchecked);
    m_BindCursorsCheckBox->setCheckState(Qt::Checked);
    m_BindMagnificationCheckBox->setCheckState(Qt::Checked);
    m_BindGeometryCheckBox->setCheckState(Qt::Checked);
    break;
  case MIDAS_BIND_GEOMETRY_WITHOUT_MAGNIFICATION:
    m_BindNoneCheckBox->setCheckState(Qt::Unchecked);
    m_BindCursorsCheckBox->setCheckState(Qt::Checked);
    m_BindMagnificationCheckBox->setCheckState(Qt::Unchecked);
    m_BindGeometryCheckBox->setCheckState(Qt::Checked);
    break;
  default:
    qWarning() << "QmitkMIDASBindWidget::SetToBindType, unrecognised type, can't set check box";
  }

  m_CurrentBindType = bindType;

  this->SetBlockSignals(false);
}

void QmitkMIDASBindWidget::OnNoneCheckBoxStateChanged(int state)
{
  this->SetBlockSignals(true);
  if (state == Qt::Checked)
  {
    m_BindCursorsCheckBox->setCheckState(Qt::Unchecked);
    m_BindMagnificationCheckBox->setCheckState(Qt::Unchecked);
    m_BindGeometryCheckBox->setCheckState(Qt::Unchecked);
  }
  else if (state == Qt::Unchecked)
  {
    // Doesn't make sense to uncheck "none", so force it back on.
    m_BindNoneCheckBox->setCheckState(Qt::Checked);
  }
  m_CurrentBindType = MIDAS_BIND_NONE;
  this->SetBlockSignals(false);
  emit BindTypeChanged(m_CurrentBindType);
}

void QmitkMIDASBindWidget::OnCursorsCheckBoxStateChanged(int state)
{
  this->SetBlockSignals(true);
  if (state == Qt::Checked)
  {
    if (m_CurrentBindType == MIDAS_BIND_MAGNIFICATION_WITHOUT_CURSORS)
    {
      m_CurrentBindType = MIDAS_BIND_MAGNIFICATION_WITH_CURSORS;
    }
    else
    {
      m_BindNoneCheckBox->setCheckState(Qt::Unchecked);
      m_BindMagnificationCheckBox->setCheckState(Qt::Unchecked);
      m_BindGeometryCheckBox->setCheckState(Qt::Unchecked);
      m_CurrentBindType = MIDAS_BIND_CURSORS;
    }
  }
  else if (state == Qt::Unchecked)
  {
    if (m_CurrentBindType == MIDAS_BIND_MAGNIFICATION_WITH_CURSORS)
    {
      m_CurrentBindType = MIDAS_BIND_MAGNIFICATION_WITHOUT_CURSORS;
    }
    else
    {
      m_BindNoneCheckBox->setCheckState(Qt::Checked);
      m_BindMagnificationCheckBox->setCheckState(Qt::Unchecked);
      m_BindGeometryCheckBox->setCheckState(Qt::Unchecked);
      m_CurrentBindType = MIDAS_BIND_NONE;
    }
  }
  this->SetBlockSignals(false);
  emit BindTypeChanged(m_CurrentBindType);
}

void QmitkMIDASBindWidget::OnMagnificationCheckBoxStateChanged(int state)
{
  this->SetBlockSignals(true);
  if (state == Qt::Checked)
  {
    if (m_CurrentBindType == MIDAS_BIND_CURSORS)
    {
      m_CurrentBindType = MIDAS_BIND_MAGNIFICATION_WITH_CURSORS;
    }
    else if (m_CurrentBindType == MIDAS_BIND_GEOMETRY_WITHOUT_MAGNIFICATION)
    {
      m_CurrentBindType = MIDAS_BIND_GEOMETRY_WITH_MAGNIFICATION;
    }
    else
    {
      m_BindNoneCheckBox->setCheckState(Qt::Unchecked);
      m_BindCursorsCheckBox->setCheckState(Qt::Unchecked);
      m_BindGeometryCheckBox->setCheckState(Qt::Unchecked);
      m_CurrentBindType = MIDAS_BIND_MAGNIFICATION_WITHOUT_CURSORS;
    }
  }
  else if (state == Qt::Unchecked)
  {
    if (m_CurrentBindType == MIDAS_BIND_MAGNIFICATION_WITHOUT_CURSORS)
    {
      m_BindNoneCheckBox->setCheckState(Qt::Checked);
      m_CurrentBindType = MIDAS_BIND_NONE;
    }
    else if (m_CurrentBindType == MIDAS_BIND_MAGNIFICATION_WITH_CURSORS)
    {
      m_CurrentBindType = MIDAS_BIND_CURSORS;
    }
    else if (m_CurrentBindType == MIDAS_BIND_GEOMETRY_WITH_MAGNIFICATION)
    {
      m_CurrentBindType = MIDAS_BIND_GEOMETRY_WITHOUT_MAGNIFICATION;
    }
  }
  this->SetBlockSignals(false);
  emit BindTypeChanged(m_CurrentBindType);
}

void QmitkMIDASBindWidget::OnGeometryCheckBoxStateChanged(int state)
{
  this->SetBlockSignals(true);
  if (state == Qt::Checked)
  {
    if (m_CurrentBindType == MIDAS_BIND_CURSORS)
    {
      m_CurrentBindType = MIDAS_BIND_GEOMETRY_WITHOUT_MAGNIFICATION;
    }
    else if (m_CurrentBindType == MIDAS_BIND_MAGNIFICATION_WITHOUT_CURSORS)
    {
      m_BindCursorsCheckBox->setCheckState(Qt::Checked);
      m_CurrentBindType = MIDAS_BIND_GEOMETRY_WITH_MAGNIFICATION;
    }
    else if (m_CurrentBindType == MIDAS_BIND_MAGNIFICATION_WITH_CURSORS)
    {
      m_CurrentBindType = MIDAS_BIND_GEOMETRY_WITH_MAGNIFICATION;
    }
    else
    {
      // Default MIDAS behaviour is effectively "none" then "all", so turn all on.
      m_BindNoneCheckBox->setCheckState(Qt::Unchecked);
      m_BindCursorsCheckBox->setCheckState(Qt::Checked);
      m_BindMagnificationCheckBox->setCheckState(Qt::Checked);
      m_CurrentBindType = MIDAS_BIND_GEOMETRY_WITH_MAGNIFICATION;
    }
  }
  else if (state == Qt::Unchecked)
  {
    // Default MIDAS behviour is effectively "none" then "all", so revert to none.
    m_BindNoneCheckBox->setCheckState(Qt::Checked);
    m_BindCursorsCheckBox->setCheckState(Qt::Unchecked);
    m_BindMagnificationCheckBox->setCheckState(Qt::Unchecked);
    m_CurrentBindType = MIDAS_BIND_NONE;
  }
  this->SetBlockSignals(false);
  emit BindTypeChanged(m_CurrentBindType);
}
