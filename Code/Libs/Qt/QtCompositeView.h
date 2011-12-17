/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-03-03 16:43:01 +0000 (Thu, 03 Mar 2011) $
 Revision          : $Revision: 6840 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QTCOMPOSITEVIEW_H
#define QTCOMPOSITEVIEW_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include <QPaintEvent>
#include <QVector>
#include "QtFramedView.h"

class QGridLayout;

/**
 * \class QtCompositeView
 * \brief Qt base class to contain multiple QWidgets, draw a border round them,
 * and coordinate layout and re-rendering them in rows and columns.
 *
 * For example, an ortho-viewer widget could contain 3 QVTKMultipleSliceView, to
 * render the coronal, axial and sagittal slices, and also a 3D viewer to see the planes.
 *
 * Note that this class only contains a copy of the pointers to the object you are laying
 * out. So, which ever class provides the widgets will be responsible for deleting them.
 * eg. See QtOrthoView, which subclasses this one, provides 3 QVTKMultipleSliceView and
 * 1 QVTK3D view, and its the subclass QtOrthoView that deletes those objects, whereas
 * this class just shuffles them around.
 */
class NIFTKQT_WINEXPORT QtCompositeView : public QtFramedView
{

  Q_OBJECT

public:

  QtCompositeView(QWidget* parent=0);
  virtual ~QtCompositeView();

  /** Sets the layout to this number of rows and columns. It's up to the user to do this sensibly. If you have X viewers, and ask for Y rows and Z columns, the widgets are just distributed across rows first, then columns. */
  virtual void SetLayoutRowsAndColumns(int rows, int columns);

  /** Will layout all widgets in a column. If maxNumberOfItems==0, then all widgets in m_Viewers are laid out, whereas if you specify a maximum, you only get that number of widgets. */
  void SetLayoutVertical(int maxNumberOfItems=0);

  /** Will layout all widgets in a row. If maxNumberOfItems==0, then all widgets in m_Viewers are laid out, whereas if you specify a maximum, you only get that number of widgets. */
  void SetLayoutHorizontal(int maxNumberOfItems=0);

  /** Returns the current number of visible rows. */
  int GetCurrentNumbeOfRows() const;

  /** Returns the current number of visible cols. */
  int GetCurrentNumberOfColumns() const;

  /** Returns the number of widgets. */
  int GetNumberOfWidgets() const { return m_WidgetPointers.size(); }

public slots:

signals:

protected:

  /** Adds the widget to the list. The Layout is not changed, until the number of rows and columns are specified. */
  void AddWidgetToList(QWidget *widget);

  /** Removes a single widget from the list. The Layout is not changed, until the number of rows and columns are specified. */
  void RemoveWidgetFromList(QWidget *widget);

  /** Removes all widgets from the list. The Layout is not changed, until the number of rows and columns are specified. */
  void RemoveAllWidgetsFromList();

private:

  /** Deliberately prohibit copy constructor. */
  QtCompositeView(const QtCompositeView&){};

  /** Deliberately prohibit assignment. */
  void operator=(const QtCompositeView&){};

  QGridLayout *m_MainLayout;
  QVector<QWidget*> m_WidgetPointers;

};

#endif //QVTKCOMPOSITEVIEW_H
