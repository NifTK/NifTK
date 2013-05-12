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

#include "mitkMIDASEnums.h"

/**
 * \class QmitkMIDASBindWidget
 * \brief Qt Widget class to contain check boxes for none, cursors, magnification, geometry.
 */
class NIFTKMIDASGUI_EXPORT QmitkMIDASBindWidget : public QWidget, public Ui_QmitkMIDASBindWidget
{
  Q_OBJECT

public:

  /// \brief Constructs a QmitkMIDASBindWidget object.
  QmitkMIDASBindWidget(QWidget *parent = 0);

  /// \brief Destructs the QmitkMIDASBindWidget object.
  virtual ~QmitkMIDASBindWidget();

  /// \brief Creates the GUI.
  void setupUi(QWidget*);

  /// \brief Method to set the widget check-boxes to match the supplied bind type, without emmitting signals.
  void SetToBindType(int bindType);

  /// \brief Returns true if the render window layout is bound and false otherwise.
  bool IsLayoutBound() const;

  /// \brief Returns true if cursors are bound, and false otherwise.
  bool AreCursorsBound() const;

  /// \brief Returns true if the magnification is bound, and false otherwise.
  bool IsMagnificationBound() const;

  /// \brief Returns true if the geometry is bound and false otherwise.
  bool IsGeometryBound() const;

signals:

  /// \brief Indicates when the bind type has changed by user input, and not when SetToBindType is called.
  void BindTypeChanged();

protected slots:

  /// \brief Qt slot called when the "layout" checkbox is toggled.
  void OnLayoutCheckBoxToggled(bool value);

  /// \brief Qt slot called when the "cursors" checkbox is toggled.
  void OnCursorsCheckBoxToggled(bool value);

  /// \brief Qt slot called when the "magnification" checkbox is toggled.
  void OnMagnificationCheckBoxToggled(bool value);

  /// \brief Qt slot called when the "geometry" checkbox is toggled.
  void OnGeometryCheckBoxToggled(bool value);

protected:

private:

  int m_BindType;
};

#endif
