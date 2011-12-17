/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 6276 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef THUMBNAILVIEWERWIDGET_CPP
#define THUMBNAILVIEWERWIDGET_CPP

#include "ThumbnailViewerWidget.h"
#include <QWidget>
#include <QVBoxLayout>
#include <QStackedLayout>

const QString ThumbnailViewerWidget::OBJECT_NAME = QString("ThumbnailViewerWidget");

ThumbnailViewerWidget::ThumbnailViewerWidget(QWidget *parent)
{
  this->setupUi(this);
  this->setObjectName(OBJECT_NAME);
  this->frame->setContentsMargins(0, 0, 0, 0);

  m_CentralLayout = new QVBoxLayout(this->frame);        // top level layout: each widget can have 1 main layout
  m_CentralStackedLayout = new QStackedLayout();         // child layout: each widget can have multiple child layouts
  m_CentralStackedLayout->setContentsMargins(0, 0, 0, 0);
  m_CentralLayout->addLayout(m_CentralStackedLayout);
}

ThumbnailViewerWidget::~ThumbnailViewerWidget()
{
}

void ThumbnailViewerWidget::AddWidget(QWidget* widget)
{
  this->m_CentralStackedLayout->addWidget(widget);
}

void ThumbnailViewerWidget::InsertWidget(int page, QWidget* widget)
{
  this->m_CentralStackedLayout->insertWidget(page, widget);
}

QWidget* ThumbnailViewerWidget::GetWidget(int page)
{
  return this->m_CentralStackedLayout->widget(page);
}

void ThumbnailViewerWidget::SetCurrentIndex(int page)
{
  this->m_CentralStackedLayout->setCurrentIndex(page);
}
#endif
