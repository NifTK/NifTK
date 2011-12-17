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
#ifndef FIXEDANDMOVINGIMAGEWIDGET_CPP
#define FIXEDANDMOVINGIMAGEWIDGET_CPP

#include "FixedAndMovingImageWidget.h"

const QString FixedAndMovingImageWidget::OBJECT_NAME = QString("FixedAndMovingImageWidget");

FixedAndMovingImageWidget::FixedAndMovingImageWidget(QWidget *parent)
{
  this->setupUi(this);
  this->setWindowTitle(tr("Fixed And Moving Image Controller"));
  this->setObjectName(OBJECT_NAME);

  this->m_PreviousFixedImage = "";
  this->m_PreviousMovingImage = "";

  this->fixedLineEdit->setAcceptDrops(true);
  this->movingLineEdit->setAcceptDrops(true);

  connect(this->fixedLineEdit, SIGNAL(textChanged(QString)), SLOT(OnFixedImageTextChanged(QString)));
  connect(this->movingLineEdit, SIGNAL(textChanged(QString)), SLOT(OnMovingImageTextChanged(QString)));
}

FixedAndMovingImageWidget::~FixedAndMovingImageWidget()
{

}

void FixedAndMovingImageWidget::SetFixedImageText(QString text)
{
  this->fixedLineEdit->blockSignals(true);
  this->fixedLineEdit->setText(text);
  this->fixedLineEdit->blockSignals(false);
}

QString FixedAndMovingImageWidget::GetFixedImageText() const
{
  return this->fixedLineEdit->text();
}

void FixedAndMovingImageWidget::SetMovingImageText(QString text)
{
  this->movingLineEdit->blockSignals(true);
  this->movingLineEdit->setText(text);
  this->movingLineEdit->blockSignals(false);
}

QString FixedAndMovingImageWidget::GetMovingImageText() const
{
  return this->movingLineEdit->text();
}

void FixedAndMovingImageWidget::OnFixedImageTextChanged(QString text)
{
  emit FixedImageTextChanged(m_PreviousFixedImage, text);
}

void FixedAndMovingImageWidget::OnMovingImageTextChanged(QString text)
{
  emit MovingImageTextChanged(m_PreviousMovingImage, text);
}

#endif
