/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASSlidersWidget.h"
#include <QDebug>

//-----------------------------------------------------------------------------
QmitkMIDASSlidersWidget::QmitkMIDASSlidersWidget(QWidget *parent)
{
  this->setupUi(this);
}


//-----------------------------------------------------------------------------
QmitkMIDASSlidersWidget::~QmitkMIDASSlidersWidget()
{

}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::setupUi(QWidget* parent)
{
  Ui_QmitkMIDASSlidersWidget::setupUi(parent);
  m_SliceSelectionWidget->setDecimals(0);
  m_SliceSelectionWidget->setTickInterval(1.0);
  m_SliceSelectionWidget->setSingleStep(1.0);
  m_MagnificationFactorWidget->setDecimals(2);
  m_MagnificationFactorWidget->setTickInterval(1.0);
  m_MagnificationFactorWidget->setSingleStep(1.0);
  m_TimeSelectionWidget->setDecimals(0);
  m_TimeSelectionWidget->setTickInterval(1.0);
  m_TimeSelectionWidget->setSingleStep(1.0);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSlidersWidget::BlockSignals(bool block)
{
  bool wasBlocked = m_MagnificationFactorWidget->signalsBlocked();
  m_MagnificationFactorWidget->blockSignals(block);
  m_SliceSelectionWidget->blockSignals(block);
  m_TimeSelectionWidget->blockSignals(block);
  return wasBlocked;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetEnabled(bool enabled)
{
  m_MagnificationFactorWidget->setEnabled(enabled);
  m_SliceSelectionWidget->setEnabled(enabled);
  m_TimeSelectionWidget->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetSliceTracking(bool isTracking)
{
  m_SliceSelectionWidget->setTracking(isTracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetMagnificationTracking(bool isTracking)
{
  m_MagnificationFactorWidget->setTracking(isTracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetTimeTracking(bool isTracking)
{
  m_TimeSelectionWidget->setTracking(isTracking);
}
