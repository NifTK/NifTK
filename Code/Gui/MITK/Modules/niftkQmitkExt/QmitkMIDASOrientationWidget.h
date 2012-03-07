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

/**
 * \class QmitkMIDASOrientationWidget
 * \brief Qt Widget class to contain radio buttons for axial, coronal, sagittal.
 */
class QmitkMIDASOrientationWidget : public QWidget, public Ui_QmitkMIDASOrientationWidget
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  QmitkMIDASOrientationWidget(QWidget *parent = 0);

  /** Destructor. */
  ~QmitkMIDASOrientationWidget();

  /// \brief Creates the GUI.
  void setupUi(QWidget*);

  /// \brief Calls setBlockSignals(block) on all contained widgets.
  void SetBlockSignals(bool block);

  /// \brief Calls setEnabled(enabled) on all contained widgets.
  void SetEnabled(bool enabled);

signals:

protected slots:

protected:

private:

};

#endif
