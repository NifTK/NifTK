/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-05 18:07:46 +0000 (Mon, 05 Dec 2011) $
 Revision          : $Revision: 7922 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
void QmitkMIDASSlidersWidget::SetBlockSignals(bool block)
{
  m_MagnificationFactorWidget->blockSignals(block);
  m_SliceSelectionWidget->blockSignals(block);
  m_TimeSelectionWidget->blockSignals(block);
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
