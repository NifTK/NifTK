/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-12-04 08:02:40 +0000 (Sat, 04 Dec 2010) $
 Revision          : $Revision: 6840 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef DATASETVIEWLISTWIDGET_CPP
#define DATASETVIEWLISTWIDGET_CPP

#include "DataSetViewListWidget.h"
#include <QListWidgetItem>
#include <QMouseEvent>
#include <QApplication>

DataSetViewListWidget::DataSetViewListWidget(QWidget* parent)
: QListWidget(parent)
{

}

DataSetViewListWidget::~DataSetViewListWidget()
{

}

QStringList DataSetViewListWidget::GetCurrentlySelectedLabels()
{
  QList<QListWidgetItem*> items = this->selectedItems();

  QStringList itemsList;
  itemsList.append("");

  if(items.isEmpty())
  {
    itemsList.clear();
    return itemsList;
  }

  itemsList.clear();
  for(int i = 0; i < items.size(); i++)
  {
    itemsList.append(items.at(i)->text());
  }
  return itemsList;
}

QStringList DataSetViewListWidget::GetAllLabels()
{
  QStringList itemsList;
  for (int i = 0; i < this->count(); i++)
  {
    itemsList.append(this->item(i)->text());
  }
  return itemsList;
}

void DataSetViewListWidget::mousePressEvent(QMouseEvent *event)
{
  if (event->button() == Qt::LeftButton)
  {
    m_StartPoint = event->pos();
  }
  QListWidget::mousePressEvent(event);
}

void DataSetViewListWidget::mouseMoveEvent(QMouseEvent *event)
{
  if (event->buttons() & Qt::LeftButton)
  {
    int distance = (event->pos() - m_StartPoint).manhattanLength();
    if (distance >= QApplication::startDragDistance())
    {
      this->startDrag();
    }
  }
}

void DataSetViewListWidget::startDrag()
{

  /**
   * Matt: I want this to be in the order they appear on screen.
   * If you just do this->getCurrentlySelectedLabels(), the order is not guaranteed.
   */
  QStringList itemsList;
  for (int i = 0; i < count(); i++)
  {
    if (this->item(i)->isSelected())
    {
      itemsList.push_back(this->item(i)->text());
    }
  }

  QString fileNameList;
  for (int i = 0; i < itemsList.size(); i++)
  {
    fileNameList.append(itemsList[i]);
    if (i < itemsList.size()-1)
    {
      fileNameList.append("|");
    }
  }

  if (itemsList.size() > 0)
  {
    QMimeData *mimeData = new QMimeData;
    mimeData->setText(fileNameList);

    QDrag *drag = new QDrag(this);
    drag->setMimeData(mimeData);
    drag->setPixmap(QPixmap(":/images/icon.png"));

    drag->exec(Qt::CopyAction | Qt::MoveAction);
  }
}

#endif
