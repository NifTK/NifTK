/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 6322 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef SLICEORIENTATIONWIDGET_CPP
#define SLICEORIENTATIONWIDGET_CPP

#include "SliceOrientationWidget.h"

const QString SliceOrientationWidget::OBJECT_NAME = QString("SliceOrientationWidget");

SliceOrientationWidget::SliceOrientationWidget(QWidget *parent)
{
  this->setupUi(this);
  this->setObjectName(OBJECT_NAME);
  this->coronalRadioButton->setChecked(true);
  this->coronalRadioButton->setToolTip(tr("Coronal view"));
  this->axialRadioButton->setToolTip(tr("Axial view"));
  this->sagittalRadioButton->setToolTip(tr("Sagittal view"));
  m_PreviousSliceOrientation = this->GetSliceOrientation();

  connect(this->axialRadioButton, SIGNAL(pressed()), this, SLOT(StorePreviousSliceOrientation()));
  connect(this->coronalRadioButton, SIGNAL(pressed()), this, SLOT(StorePreviousSliceOrientation()));
  connect(this->sagittalRadioButton, SIGNAL(pressed()), this, SLOT(StorePreviousSliceOrientation()));
  connect(this->axialRadioButton, SIGNAL(released()), this, SLOT(TriggerSignal()));
  connect(this->coronalRadioButton, SIGNAL(released()), this, SLOT(TriggerSignal()));
  connect(this->sagittalRadioButton, SIGNAL(released()), this, SLOT(TriggerSignal()));

}

SliceOrientationWidget::~SliceOrientationWidget()
{
}

void SliceOrientationWidget::SetSliceOrientation(ViewerSliceOrientation orientation)
{
  m_PreviousSliceOrientation = this->GetSliceOrientation();

  if (orientation == VIEWER_ORIENTATION_AXIAL)
  {
    this->axialRadioButton->blockSignals(true);
    this->axialRadioButton->setChecked(true);
    this->axialRadioButton->blockSignals(false);
  }
  else if (orientation == VIEWER_ORIENTATION_CORONAL)
  {
    this->coronalRadioButton->blockSignals(true);
    this->coronalRadioButton->setChecked(true);
    this->coronalRadioButton->blockSignals(false);
  }
  else if (orientation == VIEWER_ORIENTATION_SAGITTAL)
  {
    this->sagittalRadioButton->blockSignals(true);
    this->sagittalRadioButton->setChecked(true);
    this->sagittalRadioButton->blockSignals(false);
  }
}

ViewerSliceOrientation SliceOrientationWidget::GetSliceOrientation() const
{
  if (this->axialRadioButton->isChecked())
  {
    return VIEWER_ORIENTATION_AXIAL;
  }
  else if (this->coronalRadioButton->isChecked())
  {
    return VIEWER_ORIENTATION_CORONAL;
  }
  else if (this->sagittalRadioButton->isChecked())
  {
    return VIEWER_ORIENTATION_SAGITTAL;
  }
  else
  {
    return VIEWER_ORIENTATION_UNKNOWN;
  }
}

QString SliceOrientationWidget::GetLabelText(ViewerSliceOrientation orientation)
{
  if (orientation == VIEWER_ORIENTATION_AXIAL)
  {
    return this->axialRadioButton->text();
  }
  else if (orientation == VIEWER_ORIENTATION_CORONAL)
  {
    return this->coronalRadioButton->text();
  }
  else if (orientation == VIEWER_ORIENTATION_SAGITTAL)
  {
    return this->sagittalRadioButton->text();
  }
  else
  {
    return "UNKNOWN";
  }
}

void SliceOrientationWidget::StorePreviousSliceOrientation()
{
  m_PreviousSliceOrientation = this->GetSliceOrientation();
}

void SliceOrientationWidget::TriggerSignal()
{
  QString oldText = this->GetLabelText(m_PreviousSliceOrientation);
  QString newText = this->GetLabelText(this->GetSliceOrientation());

  emit SliceOrientationChanged(m_PreviousSliceOrientation, this->GetSliceOrientation(), oldText, newText);
}

#endif
