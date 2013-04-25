/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASOrientationWidget.h"
#include <QtDebug>
#include <QButtonGroup>

#include <mitkLogMacros.h>

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
  m_MultiWindowComboBox->addItem("other");
  m_MultiWindowComboBox->addItem("2x2");
  m_MultiWindowComboBox->addItem("3D");
  m_MultiWindowComboBox->addItem("3H");
  m_MultiWindowComboBox->addItem("3V");

  m_AxialWindowRadioButton->setChecked(true);

  m_ButtonGroup = new QButtonGroup(parent);
  m_ButtonGroup->addButton(m_AxialWindowRadioButton);
  m_ButtonGroup->addButton(m_SagittalWindowRadioButton);
  m_ButtonGroup->addButton(m_CoronalWindowRadioButton);
  m_ButtonGroup->addButton(m_3DWindowRadioButton);
  m_ButtonGroup->addButton(m_MultiWindowRadioButton);

  connect(m_AxialWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnAxialWindowRadioButtonToggled(bool)));
  connect(m_SagittalWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnSagittalWindowRadioButtonToggled(bool)));
  connect(m_CoronalWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnCoronalWindowRadioButtonToggled(bool)));
  connect(m_3DWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(On3DWindowRadioButtonToggled(bool)));
  connect(m_MultiWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnMultiWindowRadioButtonToggled(bool)));
  connect(m_MultiWindowComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnMultiWindowComboBoxIndexChanged(int)));
}

bool QmitkMIDASOrientationWidget::BlockSignals(bool block)
{
  bool wasBlocked = m_AxialWindowRadioButton->signalsBlocked();
  m_AxialWindowRadioButton->blockSignals(block);
  m_CoronalWindowRadioButton->blockSignals(block);
  m_SagittalWindowRadioButton->blockSignals(block);
  m_3DWindowRadioButton->blockSignals(block);
  m_MultiWindowRadioButton->blockSignals(block);
  m_MultiWindowComboBox->blockSignals(block);
  return wasBlocked;
}

void QmitkMIDASOrientationWidget::SetEnabled(bool enabled)
{
  m_AxialWindowRadioButton->setEnabled(enabled);
  m_SagittalWindowRadioButton->setEnabled(enabled);
  m_CoronalWindowRadioButton->setEnabled(enabled);
  m_3DWindowRadioButton->setEnabled(enabled);
  m_MultiWindowRadioButton->setEnabled(enabled);
  m_MultiWindowComboBox->setEnabled(enabled);
}

void QmitkMIDASOrientationWidget::SetToView(MIDASView view)
{
  if (view == m_CurrentView)
  {
    // Nothing to do.
    return;
  }

  bool wasBlocked = this->BlockSignals(true);

  switch(view)
  {
  case MIDAS_VIEW_AXIAL:
    this->m_AxialWindowRadioButton->setChecked(true);
    break;
  case MIDAS_VIEW_SAGITTAL:
    this->m_SagittalWindowRadioButton->setChecked(true);
    break;
  case MIDAS_VIEW_CORONAL:
    this->m_CoronalWindowRadioButton->setChecked(true);
    break;
  case MIDAS_VIEW_3D:
    this->m_MultiWindowRadioButton->setChecked(true);
    this->m_MultiWindowComboBox->setCurrentIndex(2);
    break;
  case MIDAS_VIEW_ORTHO:
    this->m_MultiWindowRadioButton->setChecked(true);
    this->m_MultiWindowComboBox->setCurrentIndex(1);
    break;
  case MIDAS_VIEW_3H:
    this->m_MultiWindowRadioButton->setChecked(true);
    this->m_MultiWindowComboBox->setCurrentIndex(3);
    break;
  case MIDAS_VIEW_3V:
    this->m_MultiWindowRadioButton->setChecked(true);
    this->m_MultiWindowComboBox->setCurrentIndex(4);
    break;
  default:
    qWarning() << "QmitkMIDASOrientationWidget::SetToView, unrecognised view, can't set radio button";
    break;
  }

  m_CurrentView = view;

  this->BlockSignals(wasBlocked);
}

void QmitkMIDASOrientationWidget::OnAxialWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    bool wasBlocked = this->BlockSignals(true);
    m_MultiWindowComboBox->setCurrentIndex(0);
    m_CurrentView = MIDAS_VIEW_AXIAL;
    this->BlockSignals(wasBlocked);

    emit ViewChanged(m_CurrentView);
  }
}

void QmitkMIDASOrientationWidget::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    bool wasBlocked = this->BlockSignals(true);
    m_MultiWindowComboBox->setCurrentIndex(0);
    m_CurrentView = MIDAS_VIEW_SAGITTAL;
    this->BlockSignals(wasBlocked);

    emit ViewChanged(m_CurrentView);
  }
}

void QmitkMIDASOrientationWidget::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    bool wasBlocked = this->BlockSignals(true);
    m_MultiWindowComboBox->setCurrentIndex(0);
    m_CurrentView = MIDAS_VIEW_CORONAL;
    this->BlockSignals(wasBlocked);

    emit ViewChanged(m_CurrentView);
  }
}

void QmitkMIDASOrientationWidget::On3DWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    bool wasBlocked = this->BlockSignals(true);
    m_MultiWindowComboBox->setCurrentIndex(0);
    m_CurrentView = MIDAS_VIEW_3D;
    this->BlockSignals(wasBlocked);

    emit ViewChanged(m_CurrentView);
  }
}

void QmitkMIDASOrientationWidget::OnMultiWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->m_MultiWindowComboBox->setCurrentIndex(1);
  }
}

void QmitkMIDASOrientationWidget::OnMultiWindowComboBoxIndexChanged(int index)
{
  if (index == 0)
  {
    // Force it back to what it was, as selecting "other" is not-valid.
    this->SetToView(m_CurrentView);
  }
  else
  {
    if (index == 1)
    {
      this->SetToView(MIDAS_VIEW_ORTHO);
    }
    else if (index == 2)
    {
      this->SetToView(MIDAS_VIEW_3D);
    }
    else if (index == 3)
    {
      this->SetToView(MIDAS_VIEW_3H);
    }
    else if (index == 4)
    {
      this->SetToView(MIDAS_VIEW_3V);
    }

    emit ViewChanged(m_CurrentView);
  }
}
