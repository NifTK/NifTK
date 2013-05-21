/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIFiducialRegistrationWidgetDialog_h
#define QmitkIGIFiducialRegistrationWidgetDialog_h

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
