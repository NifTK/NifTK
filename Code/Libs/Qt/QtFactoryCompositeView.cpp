/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-24 12:19:23 +0100 (Sun, 24 Jul 2011) $
 Revision          : $Revision: 6840 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QTFACTORYCOMPOSITEVIEW_CPP
#define QTFACTORYCOMPOSITEVIEW_CPP

#include "QtFactoryCompositeView.h"
#include <QWidget>
#include <cassert>

QtFactoryCompositeView::QtFactoryCompositeView(QWidget* parent)
: QtCompositeView(parent)
{
  m_Factory = NULL;
}

QtFactoryCompositeView::~QtFactoryCompositeView()
{
  for (int i = 0; i < m_Widgets.size(); i++)
  {
    if (m_Widgets[i] != NULL)
    {
      delete m_Widgets[i];
    }
  }
}

void QtFactoryCompositeView::SetQtWidgetFactory(QtWidgetFactory *factory)
{
  m_Factory = factory;
}

void QtFactoryCompositeView::SetLayoutRowsAndColumns(int rows, int columns)
{
  assert(m_Factory);

  int requiredTotalNumberOfWidgets = rows * columns;
  int currentNumberOfWidgets = m_Widgets.size();
  int numberOfWidgetsToCreate = requiredTotalNumberOfWidgets - currentNumberOfWidgets;

  if (numberOfWidgetsToCreate != 0)
  {
    if (numberOfWidgetsToCreate < 0)
    {
      int numberToRemove = abs(numberOfWidgetsToCreate);
      for (int i = requiredTotalNumberOfWidgets; i < m_Widgets.size(); i++)
      {
        if (m_Widgets[i] != NULL)
        {
          delete m_Widgets[i];
        }
      }
      m_Widgets.remove(requiredTotalNumberOfWidgets, numberToRemove);
    }
    else
    {
      for (int i = 0; i < numberOfWidgetsToCreate; i++)
      {
        QWidget* newWidget = m_Factory->CreateWidget(this);
        m_Widgets.push_back(newWidget);
      }
    }
    // This just manipulates pointers in the base class.
    this->RemoveAllWidgetsFromList();
    for (int i = 0; i < m_Widgets.size(); i++)
    {
      this->AddWidgetToList(m_Widgets[i]);
    }
  }

  // And this does the final layout
  QtCompositeView::SetLayoutRowsAndColumns(rows, columns);
}

QWidget* QtFactoryCompositeView::GetWidget(int rows, int columns)
{
  QWidget *result = NULL;
  if (rows >= 0 && rows < this->GetCurrentNumbeOfRows() && columns >= 0 && columns < this->GetCurrentNumberOfColumns())
  {
    int i = rows * this->GetCurrentNumberOfColumns() + columns;
    result = m_Widgets[i];
  }
  return result;
}

QWidget* QtFactoryCompositeView::GetWidget(int i)
{
  QWidget *result = NULL;
  if (i >= 0 && i < m_Widgets.size())
  {
    result = m_Widgets[i];
  }
  return result;
}

#endif
