/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatUploadDialog_h
#define XnatUploadDialog_h

#include <QDialog>

class QCloseEvent;
class QLabel;

class XnatUploadDialog : public QDialog
{
  Q_OBJECT

public:
  XnatUploadDialog(QWidget* parent);
  void showUploadStarting();
  void showBytesUploaded(unsigned long numBytes);
  bool wasUploadCanceled();
  void closeEvent(QCloseEvent* event);

public slots:
  bool close();

private slots:
  void cancelClicked();

private:
  bool uploadCanceled;
  QLabel* statusLabel;
};

#endif
