/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkWindowLayoutWidget.h"

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
, m_Layout(WINDOW_LAYOUT_UNKNOWN)
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
WindowLayout niftkWindowLayoutWidget::GetLayout() const
{
  return m_Layout;
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::SetLayout(WindowLayout layout)
{
  if (layout == m_Layout)
  {
    // Nothing to do.
    return;
  }

  bool wasBlocked;

  switch(layout)
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
    int layoutIndex = 0;
    while (layoutIndex < s_MultiWindowLayoutNumber && layout != s_MultiWindowLayouts[layoutIndex])
    {
      ++layoutIndex;
    }
    if (layoutIndex == s_MultiWindowLayoutNumber)
    {
      // Should not happen.
      return;
    }

    wasBlocked = m_MultiWindowRadioButton->blockSignals(true);
    m_MultiWindowRadioButton->setChecked(true);
    m_MultiWindowRadioButton->blockSignals(wasBlocked);

    wasBlocked = m_MultiWindowComboBox->blockSignals(true);
    m_MultiWindowComboBox->setCurrentIndex(layoutIndex);
    m_MultiWindowComboBox->blockSignals(wasBlocked);
    break;
  }

  m_Layout = layout;
  emit LayoutChanged(layout);
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::OnAxialWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetLayout(WINDOW_LAYOUT_AXIAL);
  }
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetLayout(WINDOW_LAYOUT_SAGITTAL);
  }
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetLayout(WINDOW_LAYOUT_CORONAL);
  }
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::On3DWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetLayout(WINDOW_LAYOUT_3D);
  }
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::OnMultiWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->SetLayout(s_MultiWindowLayouts[m_MultiWindowComboBox->currentIndex()]);
  }
}


//-----------------------------------------------------------------------------
void niftkWindowLayoutWidget::OnMultiWindowComboBoxIndexChanged(int index)
{
  m_MultiWindowRadioButton->setChecked(true);
  this->SetLayout(s_MultiWindowLayouts[index]);
}
