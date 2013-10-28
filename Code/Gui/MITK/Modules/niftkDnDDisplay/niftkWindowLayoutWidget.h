/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkWindowLayoutWidget_h
#define niftkWindowLayoutWidget_h

#include <niftkDnDDisplayExports.h>
#include "ui_niftkWindowLayoutWidget.h"

#include <mitkMIDASEnums.h>
#include <niftkDnDDisplayEnums.h>

class QButtonGroup;

/**
 * \class niftkWindowLayoutWidget
 * \brief Qt Widget class to contain four radio buttons for single window layouts (axial,
 * sagittal, coronal and 3D) and a fifth radio button and a combo box to switch to multiple
 * window layout and select it.
 *
 * By default MIDAS only provides axial, coronal and sagittal, whereas here we can use the combo box
 * for any number of layouts, and still keep a reasonably compact screen layout.
 *
 */
class niftkWindowLayoutWidget : public QWidget, private Ui_niftkWindowLayoutWidget
{
  Q_OBJECT

public:

  /// \brief Constructs a niftkWindowLayoutWidget object.
  niftkWindowLayoutWidget(QWidget *parent = 0);

  /// \brief Destructs a niftkWindowLayoutWidget object.
  virtual ~niftkWindowLayoutWidget();

  /// \brief Gets the current layout.
  WindowLayout GetLayout() const;

  /// \brief Sets the widget controls to match the supplied layout.
  void SetLayout(WindowLayout layout);

signals:

  /// \brief Indicates when the layout has changed.
  void LayoutChanged(WindowLayout layout);

protected slots:

  /// \brief Called when the axial window radio button is toggled.
  void OnAxialWindowRadioButtonToggled(bool checked);

  /// \brief Called when the sagittal window radio button is toggled.
  void OnSagittalWindowRadioButtonToggled(bool checked);

  /// \brief Called when the coronal window radio button is toggled.
  void OnCoronalWindowRadioButtonToggled(bool checked);

  /// \brief Called when the 3D window radio button is toggled.
  void On3DWindowRadioButtonToggled(bool checked);

  /// \brief Called when the multiple window radio button is toggled.
  void OnMultiWindowRadioButtonToggled(bool checked);

  /// \brief Called when a window layout is selected in the the combo box.
  void OnMultiWindowComboBoxIndexChanged(int index);

private:

  /// \brief Stores the currently selected window layout.
  WindowLayout m_Layout;

  /// \brief Stores the multiple window layouts in the same order as the combo box.
  static WindowLayout s_MultiWindowLayouts[];
  static int const s_MultiWindowLayoutNumber;
};

#endif
