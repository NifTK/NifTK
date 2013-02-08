/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatUploadManager_h
#define XnatUploadManager_h

#include "XnatRestWidgetsExports.h"

#include <QObject>
#include <QString>
#include <QStringList>

extern "C"
{
#include "XnatRest.h"
}

class XnatSettings;
class XnatTreeView;
class XnatUploadDialog;


class XnatRestWidgets_EXPORT XnatUploadManager : public QObject
{
  Q_OBJECT

public:
  XnatUploadManager(XnatTreeView* xnatTreeView);

  void setSettings(XnatSettings* settings);

  void uploadSavedData(const QString& dir);

public slots:
  void uploadFiles();

private slots:
  void zipFiles();
  void startUpload();
  void uploadData();

private:
  bool getFilenames();
  bool startFileUpload(const QString& zipFilename);
  void refreshRows();

  XnatUploadDialog* uploadDialog;

  QString currDir;
  QStringList userFilePaths;
  QString zipFilename;
  XnatRestAsynStatus finished;
  unsigned long totalBytes;

  XnatTreeView* xnatTreeView;
  XnatSettings* settings;
};

#endif
