/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASOrientationWidget.h"

MIDASView QmitkMIDASOrientationWidget::s_MultiWindowViews[] = {
  MIDAS_VIEW_ORTHO,
  MIDAS_VIEW_3H,
  MIDAS_VIEW_3V,
  MIDAS_VIEW_COR_SAG_H,
  MIDAS_VIEW_COR_SAG_V,
  MIDAS_VIEW_COR_AX_H,
  MIDAS_VIEW_COR_AX_V,
  MIDAS_VIEW_SAG_AX_H,
  MIDAS_VIEW_SAG_AX_V
};

int const QmitkMIDASOrientationWidget::s_MultiWindowViewNumber = sizeof(s_MultiWindowViews) / sizeof(MIDASView);


//-----------------------------------------------------------------------------
QmitkMIDASOrientationWidget::QmitkMIDASOrientationWidget(QWidget *parent)
: m_View(MIDAS_VIEW_UNKNOWN)
{
  setupUi(this);
}


//-----------------------------------------------------------------------------
QmitkMIDASOrientationWidget::~QmitkMIDASOrientationWidget()
{

}


//-----------------------------------------------------------------------------
void QmitkMIDASOrientationWidget::setupUi(QWidget* parent)
{
  Ui_QmitkMIDASOrientationWidget::setupUi(parent);

  m_MultiWindowComboBox->addItem("2x2");
  m_MultiWindowComboBox->addItem("3H");
  m_MultiWindowComboBox->addItem("3V");
  m_MultiWindowComboBox->addItem("cor sag H");
  m_MultiWindowComboBox->addItem("cor sag V");
  m_MultiWindowComboBox->addItem("cor ax H");
  m_MultiWindowComboBox->addItem("cor ax V");
  m_MultiWindowComboBox->addItem("sag ax H");
  m_MultiWindowComboBox->addItem("sag ax V");

  m_AxialWindowRadioButton->setChecked(true);

  connect(m_AxialWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnAxialWindowRadioButtonToggled(bool)));
  connect(m_SagittalWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnSagittalWindowRadioButtonToggled(bool)));
  connect(m_CoronalWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnCoronalWindowRadioButtonToggled(bool)));
  connect(m_3DWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(On3DWindowRadioButtonToggled(bool)));
  connect(m_MultiWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnMultiWindowRadioButtonToggled(bool)));
  connect(m_MultiWindowComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnMultiWindowComboBoxIndexChanged(int)));
}


//-----------------------------------------------------------------------------
MIDASView QmitkMIDASOrientationWidget::GetView() const
{
  return m_View;
}

//-----------------------------------------------------------------------------
void QmitkMIDASOrientationWidget::SetView(MIDASView view)
{
  if (view == m_View)
  {
    // Nothing to do.
    return;
  }

  bool wasBlocked;

  switch(view)
  {
  case MIDAS_VIEW_AXIAL:
    wasBlocked = m_AxialWindowRadioButton->blockSignals(true);
    m_AxialWindowRadioButton->setChecked(true);
    m_AxialWindowRadioButton->blockSignals(wasBlocked);
    break;
  case MIDAS_VIEW_SAGITTAL:
    wasBlocked = m_SagittalWindowRadioButton->blockSignals(true);
    m_SagittalWindowRadioButton->setChecked(true);
    m_SagittalWindowRadioButton->blockSignals(wasBlocked);
    break;
  case MIDAS_VIEW_CORONAL:
    wasBlocked = m_CoronalWindowRadioButton->blockSignals(true);
    m_CoronalWindowRadioButton->setChecked(true);
    m_CoronalWindowRadioButton->blockSignals(wasBlocked);
    break;
  case MIDAS_VIEW_3D:
    wasBlocked = m_3DWindowRadioButton->blockSignals(true);
    m_3DWindowRadioButton->setChecked(true);
    m_3DWindowRadioButton->blockSignals(wasBlocked);
    break;
  default:
    int viewIndex = 0;
    while (viewIndex < s_MultiWindowViewNumber && view != s_MultiWindowViews[viewIndex])
    {
      ++viewIndex;
    }
    if (viewIndex == s_MultiWindowViewNumber)
    {
      // Should not happen.
      return;
    }

    wasBlocked = m_MultiWindowRadioButton->blockSignals(true);
    m_MultiWindowRadioButton->setChecked(true);
    m_MultiWindowRadioButton->blockSignals(wasBlocked);

    wasBlocked = m_MultiWindowComboBox->blockSignals(true);
    m_MultiWindowComboBox->setCurrentIndex(viewIndex);
    m_MultiWindowComboBox->blockSignals(wasBlocked);
    break;
  }

  m_View = view;
  emit ViewChanged(view);
}


//-----------------------------------------------------------------------------
void QmitkMIDASOrientationWidget::OnAxialWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetView(MIDAS_VIEW_AXIAL);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASOrientationWidget::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetView(MIDAS_VIEW_SAGITTAL);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASOrientationWidget::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetView(MIDAS_VIEW_CORONAL);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASOrientationWidget::On3DWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetView(MIDAS_VIEW_3D);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASOrientationWidget::OnMultiWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetView(s_MultiWindowViews[m_MultiWindowComboBox->currentIndex()]);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASOrientationWidget::OnMultiWindowComboBoxIndexChanged(int index)
{
  m_MultiWindowRadioButton->setChecked(true);
  this->SetView(s_MultiWindowViews[index]);
}
