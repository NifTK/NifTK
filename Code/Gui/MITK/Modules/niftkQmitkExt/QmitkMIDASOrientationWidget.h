/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKMIDASORIENTATIONWIDGET_H
#define QMITKMIDASORIENTATIONWIDGET_H

#include "ui_QmitkMIDASOrientationWidget.h"
#include "mitkMIDASEnums.h"
#include <niftkQmitkExtExports.h>

class QButtonGroup;

/**
 * \class QmitkMIDASOrientationWidget
 * \brief Qt Widget class to contain radio buttons for axial, coronal, sagittal, and a combo box for any
 * other layouts of interest.
 *
 * By default MIDAS only provides axial, coronal and sagittal, whereas here we can use the combo box
 * for any number of layouts, and still keep a reasonably compact screen layout.
 *
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASOrientationWidget : public QWidget, public Ui_QmitkMIDASOrientationWidget
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  QmitkMIDASOrientationWidget(QWidget *parent = 0);
  ~QmitkMIDASOrientationWidget();

  /// \brief Creates the GUI, called from within constructor.
  void setupUi(QWidget*);

  /// \brief Calls blockSignals(block) on all contained widgets.
  bool BlockSignals(bool block);

  /// \brief Calls setEnabled(enabled) on all contained widgets.
  void SetEnabled(bool enabled);

  /// \brief Method to set the widget check-boxes to match the supplied view.
  void SetToView(MIDASView view);

signals:

  /// \brief Indicates when the view has changed.
  void ViewChanged(MIDASView view);

protected slots:

  void OnAxialRadioButtonToggled(bool checked);
  void OnCoronalRadioButtonToggled(bool checked);
  void OnSagittalRadioButtonToggled(bool checked);
  void OnOtherRadioButtonToggled(bool checked);
  void OnComboBoxIndexChanged(int i);

protected:

private:

  MIDASView m_CurrentView;
  QButtonGroup *m_ButtonGroup;
};

#endif
