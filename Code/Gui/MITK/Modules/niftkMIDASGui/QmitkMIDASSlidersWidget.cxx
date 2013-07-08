/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASSlidersWidget.h"

#include <ctkDoubleSpinBox.h>

//-----------------------------------------------------------------------------
QmitkMIDASSlidersWidget::QmitkMIDASSlidersWidget(QWidget *parent)
: QWidget(parent)
{
  this->setupUi(this);

  m_SliceIndexSliderWidget->layout()->setSpacing(3);
  m_SliceIndexSliderWidget->setDecimals(0);
  m_SliceIndexSliderWidget->setTickInterval(1.0);
  m_SliceIndexSliderWidget->setSingleStep(1.0);
  m_SliceIndexSliderWidget->spinBox()->setAlignment(Qt::AlignRight);

  m_TimeStepSliderWidget->layout()->setSpacing(3);
  m_TimeStepSliderWidget->setDecimals(0);
  m_TimeStepSliderWidget->setTickInterval(1.0);
  m_TimeStepSliderWidget->setSingleStep(1.0);
  m_TimeStepSliderWidget->spinBox()->setAlignment(Qt::AlignRight);

  m_MagnificationSliderWidget->layout()->setSpacing(3);
  m_MagnificationSliderWidget->setDecimals(2);
  m_MagnificationSliderWidget->setTickInterval(1.0);
  m_MagnificationSliderWidget->setSingleStep(1.0);
  m_MagnificationSliderWidget->spinBox()->setAlignment(Qt::AlignRight);

  connect(m_SliceIndexSliderWidget, SIGNAL(valueChanged(double)), this, SLOT(OnSliceIndexChanged(double)));
  connect(m_TimeStepSliderWidget, SIGNAL(valueChanged(double)), this, SLOT(OnTimeStepChanged(double)));
  connect(m_MagnificationSliderWidget, SIGNAL(valueChanged(double)), this, SIGNAL(MagnificationChanged(double)));
}


//-----------------------------------------------------------------------------
QmitkMIDASSlidersWidget::~QmitkMIDASSlidersWidget()
{

}


//-----------------------------------------------------------------------------
bool QmitkMIDASSlidersWidget::AreMagnificationControlsVisible() const
{
  return m_MagnificationLabel->isVisible() && m_MagnificationSliderWidget->isVisible();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetMagnificationControlsVisible(bool visible)
{
  m_MagnificationLabel->setVisible(visible);
  m_MagnificationSliderWidget->setVisible(visible);
}


//-----------------------------------------------------------------------------
int QmitkMIDASSlidersWidget::GetMaxSliceIndex() const
{
  return static_cast<int>(m_SliceIndexSliderWidget->maximum());
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetMaxSliceIndex(int maxSliceIndex)
{
  bool wasBlocked = m_SliceIndexSliderWidget->blockSignals(true);
  m_SliceIndexSliderWidget->setMaximum(maxSliceIndex);
  m_SliceIndexSliderWidget->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int QmitkMIDASSlidersWidget::GetSliceIndex() const
{
  return static_cast<int>(m_SliceIndexSliderWidget->value());
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetSliceIndex(int sliceIndex)
{
  bool wasBlocked = m_SliceIndexSliderWidget->blockSignals(true);
  m_SliceIndexSliderWidget->setValue(sliceIndex);
  m_SliceIndexSliderWidget->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int QmitkMIDASSlidersWidget::GetMaxTimeStep() const
{
  return static_cast<int>(m_TimeStepSliderWidget->maximum());
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetMaxTimeStep(int maxTimeStep)
{
  bool wasBlocked = m_TimeStepSliderWidget->blockSignals(true);
  m_TimeStepSliderWidget->setMaximum(maxTimeStep);
  m_TimeStepSliderWidget->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
int QmitkMIDASSlidersWidget::GetTimeStep() const
{
  return static_cast<int>(m_TimeStepSliderWidget->value());
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetTimeStep(int timeStep)
{
  bool wasBlocked = m_TimeStepSliderWidget->blockSignals(true);
  m_TimeStepSliderWidget->setValue(timeStep);
  m_TimeStepSliderWidget->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
double QmitkMIDASSlidersWidget::GetMinMagnification() const
{
  return m_MagnificationSliderWidget->minimum();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetMinMagnification(double minMagnification)
{
  bool wasBlocked = m_MagnificationSliderWidget->blockSignals(true);
  m_MagnificationSliderWidget->setMinimum(minMagnification);
  m_MagnificationSliderWidget->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
double QmitkMIDASSlidersWidget::GetMaxMagnification() const
{
  return m_MagnificationSliderWidget->maximum();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetMaxMagnification(double maxMagnification)
{
  bool wasBlocked = m_MagnificationSliderWidget->blockSignals(true);
  m_MagnificationSliderWidget->setMaximum(maxMagnification);
  m_MagnificationSliderWidget->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
double QmitkMIDASSlidersWidget::GetMagnification() const
{
  return m_MagnificationSliderWidget->value();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetMagnification(double magnification)
{
  bool wasBlocked = m_MagnificationSliderWidget->blockSignals(true);
  m_MagnificationSliderWidget->setValue(magnification);
  m_MagnificationSliderWidget->blockSignals(wasBlocked);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::OnSliceIndexChanged(double sliceIndex)
{
  emit SliceIndexChanged(static_cast<int>(sliceIndex));
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::OnTimeStepChanged(double timeStep)
{
  emit TimeStepChanged(static_cast<int>(timeStep));
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetSliceIndexTracking(bool isTracking)
{
  m_SliceIndexSliderWidget->setTracking(isTracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetTimeStepTracking(bool isTracking)
{
  m_TimeStepSliderWidget->setTracking(isTracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSlidersWidget::SetMagnificationTracking(bool isTracking)
{
  m_MagnificationSliderWidget->setTracking(isTracking);
}
