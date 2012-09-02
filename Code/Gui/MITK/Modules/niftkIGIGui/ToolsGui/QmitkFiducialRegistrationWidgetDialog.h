/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKFIDUCIALREGISTRATIONWIDGETDIALOG_H
#define QMITKFIDUCIALREGISTRATIONWIDGETDIALOG_H

#include <QDialog>
#include "ui_QmitkFiducialRegistrationWidgetDialog.h"

/**
 * \class QmitkFiducialRegistrationWidgetDialog
 * \brief Dialog class to wrap around the QmitkFiducialRegistrationWidget
 * to make sure we have a valid window surrounding the widget, and can run
 * modal or modalless.
 */

class QmitkFiducialRegistrationWidgetDialog : public QDialog, public Ui_QmitkFiducialRegistrationWidgetDialog
{
  Q_OBJECT

public:

  /// \brief Basic constructor.
  QmitkFiducialRegistrationWidgetDialog(QWidget *parent = 0);

  /// \brief Basic destructor.
  ~QmitkFiducialRegistrationWidgetDialog(void);

signals:

protected:

private slots:

private:

};

#endif
