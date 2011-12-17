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
#ifndef LOOKUPTABLEWIDGETWIDGET_H
#define LOOKUPTABLEWIDGETWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_LookupTableWidget.h"
#include <QWidget>
#include <QString>

class QComboBox;

/**
 * \class LookupTableWidget
 * \brief Creates a widget to select a lookup table.
 *
 * Note that the signals emitted must have globally unique names.
 * The aim is that when you adjust widgets, the signals are emitted, and
 * the only way to set the widgets are via slots.
 *
 */
class NIFTKQT_WINEXPORT LookupTableWidget : public QWidget, public Ui_LookupTableWidget {

  Q_OBJECT

public:

  /** Define this, so we can refer to it in map. */
  const static QString OBJECT_NAME;

  /** Default constructor. */
  LookupTableWidget(QWidget *parent = 0);

  /** Destructor. */
  ~LookupTableWidget();

  /** Return reference to internal combo box, so we can populate it. */
  QComboBox* GetComboBox();

  /** Get the current index. */
  int GetCurrentIndex() const;

  /** Set the current index. */
  void SetCurrentIndex(int i);

  /** Returns the text at a given index. */
  QString GetText(int i) const;

  /** Returns a list of all the items in order. */
  QStringList GetAllEntries() const;

signals:

  /** Emitted when the lookup table index changes. */
  void LookupTableIndexChanged(int oldIndex, int newIndex, QString oldName, QString newName);

private:

  LookupTableWidget(const LookupTableWidget&);  // Purposefully not implemented.
  void operator=(const LookupTableWidget&);  // Purposefully not implemented.

  int m_PreviousIndex;

private slots:

  void StorePreviousIndex();
  void TriggerLookupTableIndexChanged();
};

#endif
