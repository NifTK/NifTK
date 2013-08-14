/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatDownloadManager_h
#define XnatDownloadManager_h

#include <QObject>
#include <QScopedPointer>

class ctkXnatSettings;
class QString;
class QWidget;
class XnatDownloadDialog;
class XnatDownloadManagerPrivate;
class XnatTreeView;

class XnatDownloadManager : public QObject
{
  Q_OBJECT

public:
  XnatDownloadManager(XnatTreeView* xnatTreeView);
  virtual ~XnatDownloadManager();

  void setSettings(ctkXnatSettings* settings);

  void silentlyDownloadFile(const QString& fname, const QString& dir);
  void silentlyDownloadAllFiles(const QString& dir);

signals:
  void done();

public slots:
  void downloadFile();
  void downloadAllFiles();

private slots:
  bool startFileDownload(const QString& zipFilename);
  void startDownload();
  void startGroupDownload();
  void downloadData();
  void downloadDataAndUnzip();
  void unzipData();
  void finishDownload();
  void downloadDataBlocking(bool unzip = false);

protected:
  QScopedPointer<XnatDownloadManagerPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(XnatDownloadManager);
  Q_DISABLE_COPY(XnatDownloadManager);
};

#endif
