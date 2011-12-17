/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: 6840 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QTFRAMEDWIDGET_CPP
#define QTFRAMEDWIDGET_CPP

#include <QGridLayout>
#include "QtFramedWidget.h"

QtFramedWidget::~QtFramedWidget()
{
  if (this->m_Layout != NULL)
  {
    delete this->m_Layout;
  }

  if (this->m_Widget != NULL)
  {
    delete this->m_Widget;
  }
}

QtFramedWidget::QtFramedWidget(QWidget *parent)
: QtFramedView(parent)
{
  this->m_Widget = NULL;
  this->m_Layout = new QVBoxLayout(this);
  this->m_Layout->setContentsMargins(0, 0, 0, 0);
  this->m_Layout->setSpacing(0);
}

QWidget* QtFramedWidget::GetWidget() const
{
  return this->m_Widget;
}

void QtFramedWidget::SetWidget(QWidget *widget)
{
  this->m_Widget = widget;
  this->m_Widget->setParent(this);
  this->m_Layout->addWidget(this->m_Widget);
}

#endif
