/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : $Author$

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASBINDWIDGET_H
#define QMITKMIDASBINDWIDGET_H

#include "ui_QmitkMIDASBindWidget.h"
#include "QmitkMIDASViewEnums.h"

/**
 * \class QmitkMIDASBindWidget
 * \brief Qt Widget class to contain check boxes for none, cursors, magnification, geometry.
 */
class QmitkMIDASBindWidget : public QWidget, public Ui_QmitkMIDASBindWidget
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  QmitkMIDASBindWidget(QWidget *parent = 0);
  ~QmitkMIDASBindWidget();

  /// \brief Creates the GUI.
  void setupUi(QWidget*);

  /// \brief Calls setBlockSignals(block) on all contained widgets.
  void SetBlockSignals(bool block);

  /// \brief Calls setEnabled(enabled) on all contained widgets.
  void SetEnabled(bool enabled);

  /// \brief Method to set the widget check-boxes to match the supplied bind type, without emmitting signals.
  void SetToBindType(MIDASBindType bindType);

  /// \brief Returns true if the geometry is bound and false otherwise.
  bool IsGeometryBound() const;

  /// \brief Returns true if cursors are bound, and false otherwise.
  bool AreCursorsBound() const;

  /// \brief Returns true if magnification is bound, and false otherwise.
  bool IsMagnificationBound() const;

signals:

  /// \brief Indicates when the bind type has changed by user input, and not when SetToBindType is called.
  void BindTypeChanged(MIDASBindType bindType);

protected slots:

  /// \brief Qt slot called when the "none" checkbox is toggled.
  void OnNoneCheckBoxStateChanged(int state);

  /// \brief Qt slot called when the "cursors" checkbox is toggled.
  void OnCursorsCheckBoxStateChanged(int state);

  /// \brief Qt slot called when the "magnification" checkbox is toggled.
  void OnMagnificationCheckBoxStateChanged(int state);

  /// \brief Qt slot called when the "geometry" checkbox is toggled.
  void OnGeometryCheckBoxStateChanged(int state);

protected:

private:

  MIDASBindType m_CurrentBindType;
};

#endif
