/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkMIDASBindWidget_h
#define QmitkMIDASBindWidget_h

#include <niftkMIDASGuiExports.h>

#include "ui_QmitkMIDASBindWidget.h"

#include <mitkMIDASEnums.h>

/**
 * \class QmitkMIDASBindWidget
 * \brief Qt Widget class to contain check boxes for none, cursors, magnification, geometry.
 */
class NIFTKMIDASGUI_EXPORT QmitkMIDASBindWidget : public QWidget, private Ui_QmitkMIDASBindWidget
{
  Q_OBJECT

public:

  /// \brief Constructs a QmitkMIDASBindWidget object.
  QmitkMIDASBindWidget(QWidget *parent = 0);

  /// \brief Destructs the QmitkMIDASBindWidget object.
  virtual ~QmitkMIDASBindWidget();

  /// \brief Method to set the widget check-boxes to match the supplied bind type, without emmitting signals.
  void SetBindType(int bindType);

  /// \brief Returns true if the render window layout of the views is bound and false otherwise.
  bool AreLayoutsBound() const;

  /// \brief Returns true if the cursor of the views is bound and false otherwise.
  bool AreCursorsBound() const;

  /// \brief Returns true if the magnification of the views is bound and false otherwise.
  bool AreMagnificationsBound() const;

  /// \brief Returns true if the geometry of the views is bound and false otherwise.
  bool AreGeometriesBound() const;

signals:

  /// \brief Indicates when the bind type has changed by user input, and not when SetToBindType is called.
  void BindTypeChanged();

protected slots:

  /// \brief Qt slot called when the "layout" checkbox is toggled.
  void OnBindLayoutsCheckBoxToggled(bool value);

  /// \brief Qt slot called when the "cursors" checkbox is toggled.
  void OnBindCursorsCheckBoxToggled(bool value);

  /// \brief Qt slot called when the "magnification" checkbox is toggled.
  void OnBindMagnificationsCheckBoxToggled(bool value);

  /// \brief Qt slot called when the "geometry" checkbox is toggled.
  void OnBindGeometriesCheckBoxToggled(bool value);

protected:

private:

  int m_BindType;
};

#endif
