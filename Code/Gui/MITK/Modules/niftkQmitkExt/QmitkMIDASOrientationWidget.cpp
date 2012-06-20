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
  m_CurrentView = MIDAS_VIEW_UNKNOWN;
  setupUi(this);
}

QmitkMIDASOrientationWidget::~QmitkMIDASOrientationWidget()
{

}

void QmitkMIDASOrientationWidget::setupUi(QWidget* parent)
{
  Ui_QmitkMIDASOrientationWidget::setupUi(parent);
  m_OtherOrientationComboBox->addItem("other");
  m_OtherOrientationComboBox->addItem("2x2");
  m_OtherOrientationComboBox->addItem("3D");
  m_OtherOrientationComboBox->addItem("3H");
  m_OtherOrientationComboBox->addItem("3V");

  m_AxialRadioButton->setChecked(true);

  connect(m_AxialRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnAxialRadioButtonPressed(bool)));
  connect(m_CoronalRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnCoronalRadioButtonPressed(bool)));
  connect(m_SagittalRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnSagittalRadioButtonPressed(bool)));
  connect(m_OtherRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnOtherRadioButtonPressed(bool)));
  connect(m_OtherOrientationComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnComboBoxIndexChanged(int)));
}

void QmitkMIDASOrientationWidget::SetBlockSignals(bool block)
{
  m_AxialRadioButton->blockSignals(block);
  m_AxialLabel->blockSignals(block);
  m_CoronalRadioButton->blockSignals(block);
  m_CoronalLabel->blockSignals(block);
  m_SagittalRadioButton->blockSignals(block);
  m_SagittalLabel->blockSignals(block);
  m_OtherRadioButton->blockSignals(block);
  m_OtherOrientationComboBox->blockSignals(block);
}

void QmitkMIDASOrientationWidget::SetEnabled(bool enabled)
{
  m_AxialRadioButton->setEnabled(enabled);
  m_AxialLabel->setEnabled(enabled);
  m_CoronalRadioButton->setEnabled(enabled);
  m_CoronalLabel->setEnabled(enabled);
  m_SagittalRadioButton->setEnabled(enabled);
  m_SagittalLabel->setEnabled(enabled);
  m_OtherRadioButton->setEnabled(enabled);
  m_OtherOrientationComboBox->setEnabled(enabled);
}

void QmitkMIDASOrientationWidget::SetToView(MIDASView view)
{
  if (view == m_CurrentView)
  {
    // Nothing to do.
    return;
  }

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
    this->m_OtherRadioButton->setChecked(true);
    this->m_OtherOrientationComboBox->setCurrentIndex(1);
    break;
  case MIDAS_VIEW_3D:
    this->m_OtherRadioButton->setChecked(true);
    this->m_OtherOrientationComboBox->setCurrentIndex(2);
    break;
  case MIDAS_VIEW_3H:
    this->m_OtherRadioButton->setChecked(true);
    this->m_OtherOrientationComboBox->setCurrentIndex(3);
    break;
  case MIDAS_VIEW_3V:
    this->m_OtherRadioButton->setChecked(true);
    this->m_OtherOrientationComboBox->setCurrentIndex(4);
    break;
  default:
    qWarning() << "QmitkMIDASOrientationWidget::SetToView, unrecognised view, can't set radio button";
  }

  m_CurrentView = view;

  this->SetBlockSignals(false);
}

void QmitkMIDASOrientationWidget::OnAxialRadioButtonPressed(bool b)
{
  this->SetBlockSignals(true);
  m_OtherOrientationComboBox->setCurrentIndex(0);
  m_CurrentView = MIDAS_VIEW_AXIAL;
  this->SetBlockSignals(false);

  emit ViewChanged(m_CurrentView);
}

void QmitkMIDASOrientationWidget::OnCoronalRadioButtonPressed(bool b)
{
  this->SetBlockSignals(true);
  m_OtherOrientationComboBox->setCurrentIndex(0);
  m_CurrentView = MIDAS_VIEW_CORONAL;
  this->SetBlockSignals(false);

  emit ViewChanged(m_CurrentView);
}

void QmitkMIDASOrientationWidget::OnSagittalRadioButtonPressed(bool b)
{
  this->SetBlockSignals(true);
  m_OtherOrientationComboBox->setCurrentIndex(0);
  m_CurrentView = MIDAS_VIEW_SAGITTAL;
  this->SetBlockSignals(false);

  emit ViewChanged(m_CurrentView);
}

void QmitkMIDASOrientationWidget::OnOtherRadioButtonPressed(bool b)
{
  if (b)
  {
    this->m_OtherOrientationComboBox->setCurrentIndex(1);
  }
}

void QmitkMIDASOrientationWidget::OnComboBoxIndexChanged(int i)
{
  if (i == 0)
  {
    // Force it back to what it was, as selecting "other" is not-valid.
    this->SetToView(m_CurrentView);
  }
  else
  {
    if (i == 1)
    {
      this->SetToView(MIDAS_VIEW_ORTHO);
    }
    else if (i == 2)
    {
      this->SetToView(MIDAS_VIEW_3D);
    }
    else if (i == 3)
    {
      this->SetToView(MIDAS_VIEW_3H);
    }
    else if (i == 4)
    {
      this->SetToView(MIDAS_VIEW_3V);
    }

    emit ViewChanged(m_CurrentView);
  }
}
