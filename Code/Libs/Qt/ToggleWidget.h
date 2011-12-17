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
#ifndef TOGGLEWIDGET_H
#define TOGGLEWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_ToggleWidget.h"
#include <QStringList>
#include <QWidget>

/**
 * \class ToggleWidget
 * \brief Creates a widget to toggle between two images MIDAS style.
 *
 * Note that the signals emitted must have globally unique names.
 * The aim is that when you adjust widgets, the signals are emitted, and
 * the only way to set the widgets are via slots.
 *
 */
class NIFTKQT_WINEXPORT ToggleWidget : public QWidget, public Ui_ToggleWidget
{
  Q_OBJECT

public:

  /** Define this, so we can refer to it in map. */
  const static QString OBJECT_NAME;

  /** Default constructor. */
  ToggleWidget(QWidget *parent = 0);

  /** Destructor. */
  ~ToggleWidget();

  /** Sets the current index by number. */
  void SetCurrentIndex(int i);

  /** Sets the current index by name. */
  void SetCurrentIndex(QString name);

  /** Sets the string lists to toggle. */
  void SetDataSets(QStringList list);

  /** Inserts a string into the combo box. */
  void SetDataSet(int position, QString text);

  /** Clears the list of items. */
  void ClearList();

signals:

  /** Emitted to indicate that we have toggled */
  void DataSetToggled(int oldIndex, int newIndex, QString oldName, QString newName);

public slots:

  /** Does the actual toggling. */
  void OnButtonPressed();

private:

  ToggleWidget(const ToggleWidget&);  // Purposefully not implemented.
  void operator=(const ToggleWidget&);  // Purposefully not implemented.

  int m_PreviousIndex;

  QString m_PreviousText;
};
#endif
