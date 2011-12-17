/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-02 17:25:37 +0100 (Thu, 02 Sep 2010) $
 Revision          : $Revision: 6276 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef OPACITYWIDGET_CPP
#define OPACITYWIDGET_CPP

#include "OpacityWidget.h"

const QString OpacityWidget::OBJECT_NAME = QString("OpacityWidget");

OpacityWidget::OpacityWidget(QWidget *parent)
: DoubleSpinBoxAndSliderWidget(parent)
{
  this->SetText(tr("opacity"));
  this->setWindowTitle(tr("Opacity Controller"));
  this->setObjectName(OBJECT_NAME);
  this->SetMinimum(0);
  this->SetMaximum(1);

  connect(this->spinBox, SIGNAL(valueChanged(double)), this, SLOT(OnChangeOpacity()));
  connect(this->horizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(OnChangeOpacity()));
}

OpacityWidget::~OpacityWidget()
{

}

void OpacityWidget::SetOpacity(double value)
{
  this->SetValue(value);
}

void OpacityWidget::OnChangeOpacity()
{
  emit OpacityChanged(m_PreviousValue, this->spinBox->value());
}

#endif
