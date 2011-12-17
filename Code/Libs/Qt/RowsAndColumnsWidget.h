/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-02 17:25:37 +0100 (Thu, 02 Sep 2010) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ROWSANDCOLUMNSWIDGET_H
#define ROWSANDCOLUMNSWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_RowsAndColumnsWidget.h"
#include <QWidget>
#include <QString>

/**
 * \class RowsAndColumnsWidget
 * \brief Creates a dockable widget to select a number of rows and columns.
 *
 * Note that the signals emitted must have globally unique names.
 * The aim is that when you adjust widgets, the signals are emitted, and
 * the only way to set the widgets are via slots.
 *
 */
class NIFTKQT_WINEXPORT RowsAndColumnsWidget : public QWidget, public Ui_RowsAndColumnsWidget {

  Q_OBJECT

public:

  /** Define this, so we can refer to it in map. */
  const static QString OBJECT_NAME;

  /** Default constructor. */
  RowsAndColumnsWidget(QWidget *parent = 0);

  /** Destructor. */
  ~RowsAndColumnsWidget();

  /** Sets the number of rows. */
  void SetNumberOfRows(int i);

  /** Gets the number of rows. */
  int GetNumberOfRows() const;

  /** Sets the number of columns. */
  void SetNumberOfColumns(int i);

  /** Gets the number of columns. */
  int GetNumberOfColumns() const;

  /** Sets the minimum number of rows, defaults to 1. */
  void SetMinimumNumberOfRows(int i);

  /** Gets the minimum number of rows, defaults to 1. */
  int GetMinimumNumberOfRows() const;

  /** Sets the minimum number of columns, defaults to 1. */
  void SetMinimumNumberOfColumns(int i);

  /** Gets the minimum number of columns, defaults to 1. */
  int GetMinimumNumberOfColumns() const;

  /** Sets the maximum number of rows, defaults to 5. */
  void SetMaximumNumberOfRows(int i);

  /** Gets the maximum number of rows, defaults to 5. */
  int GetMaximumNumberOfRows() const;

  /** Sets the maximum number of columns, defaults to 5. */
  void SetMaximumNumberOfColumns(int i);

  /** Gets the maximum number of columns, defaults to 5. */
  int GetMaximumNumberOfColumns() const;

signals:

  /** Emitted when the number of rows changes. */
  void NumberOfRowsChanged(int oldNumberOfRows, int newNumberOfRows);

  /** Emitted when the number of columns changes. */
  void NumberOfColumnsChanged(int oldNumberOfColumns, int newNumberOfColumns);

private:

  RowsAndColumnsWidget(const RowsAndColumnsWidget&);  // Purposefully not implemented.
  void operator=(const RowsAndColumnsWidget&);  // Purposefully not implemented.

  int m_PreviousNumberOfRows;

  int m_PreviousNumberOfColumns;

private slots:

  void OnNumberOfRowsChanged(int i);

  void OnNumberOfColumnsChanged(int i);

};

#endif
