/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASSlidersWidget.h"

#include <mitkLogMacros.h>

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
  m_MagnificationWidget->setDecimals(2);
  m_MagnificationWidget->setTickInterval(1.0);
  m_MagnificationWidget->setSingleStep(1.0);
  m_TimeSelectionWidget->setDecimals(0);
  m_TimeSelectionWidget->setTickInterval(1.0);
  m_TimeSelectionWidget->setSingleStep(1.0);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASSlidersWidget::BlockSignals(bool block)
{
  bool wasBlocked = m_MagnificationWidget->signalsBlocked();
  m_MagnificationWidget->blockSignals(block);
  m_SliceSelectionWidget->blockSignals(block);
  m_TimeSelectionWidget->blockSignals(block);
  return wasBlocked;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetSliceTracking(bool isTracking)
{
  m_SliceSelectionWidget->setTracking(isTracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetMagnificationTracking(bool isTracking)
{
  m_MagnificationWidget->setTracking(isTracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetTimeTracking(bool isTracking)
{
  m_TimeSelectionWidget->setTracking(isTracking);
}
