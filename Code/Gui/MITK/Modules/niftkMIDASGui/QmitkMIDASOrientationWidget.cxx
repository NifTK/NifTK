/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASOrientationWidget.h"

#include <QButtonGroup>

#include <mitkLogMacros.h>

MIDASView QmitkMIDASOrientationWidget::s_Views[] = {
//  MIDAS_VIEW_UNKNOWN,
  MIDAS_VIEW_ORTHO,
  MIDAS_VIEW_3H,
  MIDAS_VIEW_3V,
  MIDAS_VIEW_SAG_COR_H,
  MIDAS_VIEW_SAG_COR_V,
  MIDAS_VIEW_AX_COR_H,
  MIDAS_VIEW_AX_COR_V,
  MIDAS_VIEW_AX_SAG_H,
  MIDAS_VIEW_AX_SAG_V
};

int const QmitkMIDASOrientationWidget::s_ViewsSize = sizeof(s_Views) / sizeof(MIDASView);

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
//  m_MultiWindowComboBox->addItem("other");
  m_MultiWindowComboBox->addItem("2x2");
  m_MultiWindowComboBox->addItem("3H");
  m_MultiWindowComboBox->addItem("3V");
  m_MultiWindowComboBox->addItem("sag cor H");
  m_MultiWindowComboBox->addItem("sag cor V");
  m_MultiWindowComboBox->addItem("ax cor H");
  m_MultiWindowComboBox->addItem("ax cor V");
  m_MultiWindowComboBox->addItem("ax sag H");
  m_MultiWindowComboBox->addItem("ax sag V");

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
    this->m_3DWindowRadioButton->setChecked(true);
    break;
  default:
    int viewIndex = 0;
    while (viewIndex < s_ViewsSize && view != s_Views[viewIndex])
    {
      ++viewIndex;
    }
    if (viewIndex != s_ViewsSize)
    {
      this->m_MultiWindowRadioButton->setChecked(true);
      this->m_MultiWindowComboBox->setCurrentIndex(viewIndex);
    }
    break;
  }

  m_CurrentView = view;

  this->BlockSignals(wasBlocked);
}

void QmitkMIDASOrientationWidget::OnAxialWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    m_CurrentView = MIDAS_VIEW_AXIAL;
    this->SetToView(m_CurrentView);
    emit ViewChanged(m_CurrentView);
  }
}

void QmitkMIDASOrientationWidget::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    m_CurrentView = MIDAS_VIEW_SAGITTAL;
    this->SetToView(m_CurrentView);
    emit ViewChanged(m_CurrentView);
  }
}

void QmitkMIDASOrientationWidget::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    m_CurrentView = MIDAS_VIEW_CORONAL;
    this->SetToView(m_CurrentView);
    emit ViewChanged(m_CurrentView);
  }
}

void QmitkMIDASOrientationWidget::On3DWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    m_CurrentView = MIDAS_VIEW_3D;
    this->SetToView(m_CurrentView);
    emit ViewChanged(m_CurrentView);
  }
}

void QmitkMIDASOrientationWidget::OnMultiWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    m_CurrentView = s_Views[m_MultiWindowComboBox->currentIndex()];
    this->SetToView(m_CurrentView);
    emit ViewChanged(m_CurrentView);
  }
}

void QmitkMIDASOrientationWidget::OnMultiWindowComboBoxIndexChanged(int index)
{
  m_CurrentView = s_Views[index];
  m_MultiWindowRadioButton->setChecked(true);
  this->SetToView(m_CurrentView);
  emit ViewChanged(m_CurrentView);
}
