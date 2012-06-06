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

QmitkMIDASSlidersWidget::QmitkMIDASSlidersWidget(QWidget *parent)
{
  this->setupUi(this);
}

QmitkMIDASSlidersWidget::~QmitkMIDASSlidersWidget()
{

}

void QmitkMIDASSlidersWidget::setupUi(QWidget* parent)
{
  Ui_QmitkMIDASSlidersWidget::setupUi(parent);
  m_MagnificationFactorWidget->setToolTip("changes the magnification of the currently selected view.");
  m_SliceSelectionWidget->setToolTip("changes the slice number of the currently selected view.");
  m_TimeSelectionWidget->SetText("time");
  m_TimeSelectionWidget->setToolTip("changes the time step number of the currently selected view.");
}

void QmitkMIDASSlidersWidget::SetSliceSliderInverted(bool inverted)
{
  m_SliceSelectionWidget->SetInverse(inverted);
}

void QmitkMIDASSlidersWidget::SetSliceSliderOffset(int offset)
{
  m_SliceSelectionWidget->SetOffset(offset);
}

void QmitkMIDASSlidersWidget::SetBlockSignals(bool block)
{
  m_MagnificationFactorWidget->SetBlockSignals(block);
  m_SliceSelectionWidget->SetBlockSignals(block);
  m_TimeSelectionWidget->SetBlockSignals(block);
}

void QmitkMIDASSlidersWidget::SetEnabled(bool enabled)
{
  m_MagnificationFactorWidget->SetEnabled(enabled);
  m_SliceSelectionWidget->SetEnabled(enabled);
  m_TimeSelectionWidget->SetEnabled(enabled);
}
