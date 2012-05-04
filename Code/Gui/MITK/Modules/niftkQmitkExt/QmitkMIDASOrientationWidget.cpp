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

#include "QmitkMIDASOrientationWidget.h"
#include <QtDebug>

QmitkMIDASOrientationWidget::QmitkMIDASOrientationWidget(QWidget *parent)
{
  setupUi(this);
}

QmitkMIDASOrientationWidget::~QmitkMIDASOrientationWidget()
{

}

void QmitkMIDASOrientationWidget::setupUi(QWidget* parent)
{
  Ui_QmitkMIDASOrientationWidget::setupUi(parent);
  m_AxialRadioButton->setChecked(true);
}

void QmitkMIDASOrientationWidget::SetBlockSignals(bool block)
{
  m_AxialRadioButton->blockSignals(block);
  m_CoronalRadioButton->blockSignals(block);
  m_OrthogonalRadioButton->blockSignals(block);
  m_SagittalRadioButton->blockSignals(block);
  m_SliceOrientationLabel->blockSignals(block);
  m_ThreeDRadioButton->blockSignals(block);
  m_ThreeHRadioButton->blockSignals(block);
  m_ThreeVRadioButton->blockSignals(block);
}

void QmitkMIDASOrientationWidget::SetEnabled(bool enabled)
{
  m_AxialRadioButton->setEnabled(enabled);
  m_CoronalRadioButton->setEnabled(enabled);
  m_OrthogonalRadioButton->setEnabled(enabled);
  m_SagittalRadioButton->setEnabled(enabled);
  m_SliceOrientationLabel->setEnabled(enabled);
  m_ThreeDRadioButton->setEnabled(enabled);
  m_ThreeHRadioButton->setEnabled(enabled);
  m_ThreeVRadioButton->setEnabled(enabled);
}

void QmitkMIDASOrientationWidget::SetToView(MIDASView view)
{
  this->SetBlockSignals(true);

  switch(view)
  {
  case MIDAS_VIEW_AXIAL:
    this->m_AxialRadioButton->setChecked(true);
    break;
  case MIDAS_VIEW_SAGITTAL:
    this->m_SagittalRadioButton->setChecked(true);
    break;
  case MIDAS_VIEW_CORONAL:
    this->m_CoronalRadioButton->setChecked(true);
    break;
  case MIDAS_VIEW_ORTHO:
    this->m_OrthogonalRadioButton->setChecked(true);
    break;
  case MIDAS_VIEW_3D:
    this->m_ThreeDRadioButton->setChecked(true);
    break;
  case MIDAS_VIEW_3H:
    this->m_ThreeHRadioButton->setChecked(true);
    break;
  case MIDAS_VIEW_3V:
    this->m_ThreeVRadioButton->setChecked(true);
    break;
  default:
    qWarning() << "QmitkMIDASOrientationWidget::SetToView, unrecognised view, can't set radio button";
  }
  this->SetBlockSignals(false);
}
