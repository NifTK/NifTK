/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkWindowLayoutWidget_p.h"

WindowLayout niftkWindowLayoutWidget::s_MultiWindowLayouts[] = {
  WINDOW_LAYOUT_ORTHO,
  WINDOW_LAYOUT_3H,
  WINDOW_LAYOUT_3V,
  WINDOW_LAYOUT_COR_SAG_H,
  WINDOW_LAYOUT_COR_SAG_V,
  WINDOW_LAYOUT_COR_AX_H,
  WINDOW_LAYOUT_COR_AX_V,
  WINDOW_LAYOUT_SAG_AX_H,
  WINDOW_LAYOUT_SAG_AX_V
};

int const niftkWindowLayoutWidget::s_MultiWindowLayoutNumber = sizeof(s_MultiWindowLayouts) / sizeof(WindowLayout);


//-----------------------------------------------------------------------------
niftkWindowLayoutWidget::niftkWindowLayoutWidget(QWidget *parent)
: QWidget(parent)
, m_WindowLayout(WINDOW_LAYOUT_UNKNOWN)
{
  this->setupUi(this);

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
niftkWindowLayoutWidget::~niftkWindowLayoutWidget()
{
}


//-----------------------------------------------------------------------------
WindowLayout niftkWindowLayoutWidget::GetWindowLayout() const
{
  return m_WindowLayout;
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::SetWindowLayout(WindowLayout windowLayout)
{
  if (windowLayout == m_WindowLayout)
  {
    // Nothing to do.
    return;
  }

  bool wasBlocked;

  switch (windowLayout)
  {
  case WINDOW_LAYOUT_AXIAL:
    wasBlocked = m_AxialWindowRadioButton->blockSignals(true);
    m_AxialWindowRadioButton->setChecked(true);
    m_AxialWindowRadioButton->blockSignals(wasBlocked);
    break;
  case WINDOW_LAYOUT_SAGITTAL:
    wasBlocked = m_SagittalWindowRadioButton->blockSignals(true);
    m_SagittalWindowRadioButton->setChecked(true);
    m_SagittalWindowRadioButton->blockSignals(wasBlocked);
    break;
  case WINDOW_LAYOUT_CORONAL:
    wasBlocked = m_CoronalWindowRadioButton->blockSignals(true);
    m_CoronalWindowRadioButton->setChecked(true);
    m_CoronalWindowRadioButton->blockSignals(wasBlocked);
    break;
  case WINDOW_LAYOUT_3D:
    wasBlocked = m_3DWindowRadioButton->blockSignals(true);
    m_3DWindowRadioButton->setChecked(true);
    m_3DWindowRadioButton->blockSignals(wasBlocked);
    break;
  default:
    int windowLayoutIndex = 0;
    while (windowLayoutIndex < s_MultiWindowLayoutNumber && windowLayout != s_MultiWindowLayouts[windowLayoutIndex])
    {
      ++windowLayoutIndex;
    }
    if (windowLayoutIndex == s_MultiWindowLayoutNumber)
    {
      // Should not happen.
      return;
    }

    wasBlocked = m_MultiWindowRadioButton->blockSignals(true);
    m_MultiWindowRadioButton->setChecked(true);
    m_MultiWindowRadioButton->blockSignals(wasBlocked);

    wasBlocked = m_MultiWindowComboBox->blockSignals(true);
    m_MultiWindowComboBox->setCurrentIndex(windowLayoutIndex);
    m_MultiWindowComboBox->blockSignals(wasBlocked);
    break;
  }

  m_WindowLayout = windowLayout;
  emit WindowLayoutChanged(windowLayout);
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::OnAxialWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  }
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  }
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  }
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::On3DWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(WINDOW_LAYOUT_3D);
  }
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::OnMultiWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetWindowLayout(s_MultiWindowLayouts[m_MultiWindowComboBox->currentIndex()]);
  }
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::OnMultiWindowComboBoxIndexChanged(int index)
{
  m_MultiWindowRadioButton->setChecked(true);
  this->SetWindowLayout(s_MultiWindowLayouts[index]);
}
