/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASORIENTATIONWIDGET_H
#define QMITKMIDASORIENTATIONWIDGET_H

#include "ui_QmitkMIDASOrientationWidget.h"
#include "QmitkMIDASViewEnums.h"
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

  /// \brief Calls setBlockSignals(block) on all contained widgets.
  void SetBlockSignals(bool block);

  /// \brief Calls setEnabled(enabled) on all contained widgets.
  void SetEnabled(bool enabled);

  /// \brief Method to set the widget check-boxes to match the supplied view.
  void SetToView(MIDASView view);

signals:

  /// \brief Indicates when the view has changed.
  void ViewChanged(MIDASView view);

protected slots:

  void OnAxialRadioButtonPressed(bool b);
  void OnCoronalRadioButtonPressed(bool b);
  void OnSagittalRadioButtonPressed(bool b);
  void OnOtherRadioButtonPressed(bool b);
  void OnComboBoxIndexChanged(int i);

protected:

private:

  MIDASView m_CurrentView;
  QButtonGroup *m_ButtonGroup;
};

#endif
