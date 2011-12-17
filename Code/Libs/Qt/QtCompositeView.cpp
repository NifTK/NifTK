/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-03-03 16:43:01 +0000 (Thu, 03 Mar 2011) $
 Revision          : $Revision: 6276 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QTCOMPOSITEVIEW_CPP
#define QTCOMPOSITEVIEW_CPP

#include <QGridLayout>
#include "QtCompositeView.h"

QtCompositeView::QtCompositeView(QWidget* parent)
: QtFramedView(parent)
{
  this->m_MainLayout = NULL;

  // Set this widget to always try expanding
  QSizePolicy expandingSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  expandingSizePolicy.setVerticalStretch(1);
  expandingSizePolicy.setHorizontalStretch(1);
  this->setSizePolicy(expandingSizePolicy);
}

QtCompositeView::~QtCompositeView()
{
  if (this->m_MainLayout != NULL)
  {
    delete this->m_MainLayout;
  }
}

int QtCompositeView::GetCurrentNumbeOfRows() const
{
  return this->m_MainLayout->rowCount();
}

int QtCompositeView::GetCurrentNumberOfColumns() const
{
  return this->m_MainLayout->columnCount();
}

void QtCompositeView::SetLayoutRowsAndColumns(int rows, int columns)
{
  // If we have a previous layout, set all widgets to invisible, remove from layout, and delete.
  if (this->m_MainLayout != NULL)
  {
    for (int r = 0; r < m_MainLayout->rowCount(); r++)
    {
      for (int c = 0; c < m_MainLayout->columnCount(); c++)
      {
        QLayoutItem *item = m_MainLayout->itemAtPosition(r, c);
        if (item != NULL)
        {
          QWidget *widget = item->widget();
          if (widget != NULL)
          {
            widget->setVisible(false);
            m_MainLayout->removeItem(m_MainLayout->itemAtPosition(r, c));
          }
        }
      }
    }
    delete m_MainLayout;
  }

  // New Grid Layout
  m_MainLayout = new QGridLayout(this);
  m_MainLayout->setContentsMargins(0, 0, 0, 0);
  m_MainLayout->setHorizontalSpacing(0);
  m_MainLayout->setVerticalSpacing(0);

  // 4. Allocate widgets along rows, then cols.
  int counter = 0;
  for (int r = 0; r < rows; r++)
  {
    for (int c = 0; c < columns; c++)
    {
      if (counter < m_WidgetPointers.size())
      {
        m_MainLayout->addWidget(m_WidgetPointers[counter], r, c);
        m_MainLayout->setRowStretch(r, 1);
        m_MainLayout->setColumnStretch(c, 1);
        m_WidgetPointers[counter]->setVisible(true);
        counter++;
      }
    }
  } // end for
}

void QtCompositeView::SetLayoutVertical(int maxNumberOfItems)
{
  int numberOfViewers = m_WidgetPointers.size();
  this->SetLayoutRowsAndColumns(numberOfViewers, 1);
}

void QtCompositeView::SetLayoutHorizontal(int maxNumberOfItems)
{
  int numberOfViewers = m_WidgetPointers.size();
  this->SetLayoutRowsAndColumns(1, numberOfViewers);
}

void QtCompositeView::AddWidgetToList(QWidget *widget)
{
  this->m_WidgetPointers.push_back(widget);
}

void QtCompositeView::RemoveWidgetFromList(QWidget *widget)
{
  for (int i = 0; i < m_WidgetPointers.size(); i++)
  {
    if (m_WidgetPointers[i] == widget)
    {
      m_WidgetPointers.remove(i);
    }
  }
}

void QtCompositeView::RemoveAllWidgetsFromList()
{
  this->m_WidgetPointers.clear();
}

#endif
