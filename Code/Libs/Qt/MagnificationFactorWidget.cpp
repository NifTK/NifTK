/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-02 17:25:37 +0100 (Thu, 02 Sep 2010) $
 Revision          : $Revision: 7921 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MAGNIFICATIONFACTORWIDGET_CPP
#define MAGNIFICATIONFACTORWIDGET_CPP

#include "MagnificationFactorWidget.h"

const QString MagnificationFactorWidget::OBJECT_NAME = QString("MagnificationFactorWidget");

MagnificationFactorWidget::MagnificationFactorWidget(QWidget *parent)
: IntegerSpinBoxAndSliderWidget(parent)
{
  this->SetText(tr("magnification"));
  this->setWindowTitle(tr("Magnificaton Factor Controller"));
  this->setToolTip(tr("Image Magnification"));
  this->setObjectName(OBJECT_NAME);
  this->SetMinimum(-5);
  this->SetMaximum(20);
}

MagnificationFactorWidget::~MagnificationFactorWidget()
{

}

void MagnificationFactorWidget::SetMagnificationFactor(int value)
{
  this->SetValue(value);
}

#endif
