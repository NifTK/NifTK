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

#ifndef QMITKMIDASSLIDERSWIDGET_H
#define QMITKMIDASSLIDERSWIDGET_H

#include "ui_QmitkMIDASSlidersWidget.h"
#include <niftkQmitkExtExports.h>

/**
 * \class QmitkMIDASSlidersWidget
 * \brief Qt Widget class to contain sliders for slice, time and magnification.
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASSlidersWidget : public QWidget, public Ui_QmitkMIDASSlidersWidget
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  QmitkMIDASSlidersWidget(QWidget *parent = 0);

  /** Destructor. */
  ~QmitkMIDASSlidersWidget();

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
