/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkMIDASOrientationWidget_h
#define QmitkMIDASOrientationWidget_h

#include <niftkMIDASGuiExports.h>
#include "ui_QmitkMIDASOrientationWidget.h"
#include "mitkMIDASEnums.h"

class QButtonGroup;

/**
 * \class QmitkMIDASOrientationWidget
 * \brief Qt Widget class to contain four radio buttons for single window layouts (axial,
 * sagittal, coronal and 3D) and a fifth radio button and a combo box to switch to multiple
 * window layout and select it.
 *
 * By default MIDAS only provides axial, coronal and sagittal, whereas here we can use the combo box
 * for any number of layouts, and still keep a reasonably compact screen layout.
 *
 */
class NIFTKMIDASGUI_EXPORT QmitkMIDASOrientationWidget : public QWidget, public Ui_QmitkMIDASOrientationWidget
{
  Q_OBJECT

public:

  /// \brief Constructs a QmitkMIDASOrientationWidget object.
  QmitkMIDASOrientationWidget(QWidget *parent = 0);

  /// \brief Destructs a QmitkMIDASOrientationWidget object.
  virtual ~QmitkMIDASOrientationWidget();

  /// \brief Creates the GUI, called from within constructor.
  void setupUi(QWidget*);

  /// \brief Calls setEnabled(enabled) on all contained widgets.
  void SetEnabled(bool enabled);

  /// \brief Method to set the widget check-boxes to match the supplied view.
  void SetView(MIDASView view);

signals:

  /// \brief Indicates when the view has changed.
  void ViewChanged(MIDASView view);

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
  MIDASView m_CurrentView;

  /// \brief Stores the multiple window layouts in the same order as the combo box.
  static MIDASView s_MultiWindowViews[];
  static int const s_MultiWindowViewNumber;
};

#endif
