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
#ifndef DATASETVIEWLISTWIDGET_H
#define DATASETVIEWLISTWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include <QWidget>
#include <QListWidget>
#include <QPoint>
#include <QStringList>

class QMouseEvent;

/**
 * \class DataSetViewListWidget
 * \brief Subclass of QListWidget to override the drag events to send out a list of selected filenames.
 */
class NIFTKQT_WINEXPORT DataSetViewListWidget : public QListWidget {

public:

  /** Constructor.  */
  DataSetViewListWidget(QWidget* parent = 0);

  /** Destructor. */
  ~DataSetViewListWidget();

  /** Returns a copy of the currently selected label(s) (eg. Filename(s)). */
  QStringList GetCurrentlySelectedLabels();

  /** Returns a copy of all the selected label(s) (eg. Filenames(s)). */
  QStringList GetAllLabels();

protected:

  /** Override base class method to respond to mouse events. */
  void mousePressEvent(QMouseEvent *event);

  /** Override base class method to respond to mouse events. */
  void mouseMoveEvent(QMouseEvent *event);

private:

  DataSetViewListWidget(const DataSetViewListWidget&);  // Not implemented.
  void operator=(const DataSetViewListWidget&);  // Not implemented.

  /** Function to initiate a drag event. */
  void startDrag();

  QPoint m_StartPoint;
};
#endif
