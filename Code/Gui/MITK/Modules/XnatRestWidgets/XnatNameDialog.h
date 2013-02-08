/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatNameDialog_h
#define XnatNameDialog_h

#include "XnatRestWidgetsExports.h"

#include <QDialog>

class QLineEdit;

class XnatRestWidgets_EXPORT XnatNameDialog : public QDialog
{
  Q_OBJECT

public:
  XnatNameDialog(QWidget* p, const QString& kind, const QString& parentName);
  const QString getNewName();

private slots:
  void accept();

private:
  QLineEdit* nameEdit;
  QString newName;
};

#endif
