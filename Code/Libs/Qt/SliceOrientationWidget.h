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
#ifndef SLICEORIENTATIONWIDGET_H
#define SLICEORIENTATIONWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_SliceOrientationWidget.h"
#include <QWidget>
#include <QString>
#include "vtkEnums.h"

/**
 * \class SliceOrientationWidget
 * \brief Creates a widget to select the slice orientation.
 *
 * Note that the signals emitted must have globally unique names.
 * The aim is that when you adjust widgets, the signals are emitted, and
 * the only way to set the widgets are via slots.
 *
 */
class NIFTKQT_WINEXPORT SliceOrientationWidget : public QWidget, public Ui_SliceOrientationWidget {

  Q_OBJECT

public:

  /** Define this, so we can refer to it in map. */
  const static QString OBJECT_NAME;

  /** Default constructor. */
  SliceOrientationWidget(QWidget *parent = 0);

  /** Destructor. */
  ~SliceOrientationWidget();

  /** Sets the current slice orientation. */
  void SetSliceOrientation(ViewerSliceOrientation orientation);

  /** Gets the current slice orientation. */
  ViewerSliceOrientation GetSliceOrientation() const;

signals:

  /** Emitted when the slice orientation changes. */
  void SliceOrientationChanged(int oldSliceOrientation, int newSliceOrientation, QString oldName, QString newName);

private:

  SliceOrientationWidget(const SliceOrientationWidget&);  // Purposefully not implemented.
  void operator=(const SliceOrientationWidget&);  // Purposefully not implemented.

  ViewerSliceOrientation m_PreviousSliceOrientation;

  /** Returns the label text from the radio button for the given orientation. */
  QString GetLabelText(ViewerSliceOrientation orientation);

private slots:

  void StorePreviousSliceOrientation();
  void TriggerSignal();
};

#endif
