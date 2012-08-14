#ifndef XnatDownloadManager_h
#define XnatDownloadManager_h

#include <QObject>

extern "C"
{
#include "XnatRest.h"
}

class QString;
class XnatBrowserWidget;
class XnatSettings;
class XnatDownloadDialog;

class XnatDownloadManager : public QObject
{
  Q_OBJECT

public:
  XnatDownloadManager(XnatBrowserWidget* b);
  void downloadFile(const QString& fname);
  void downloadAllFiles();
  void silentlyDownloadFile(const QString& fname, const QString& dir);

signals:
  void done();

private slots:
  void startDownload();
  void startGroupDownload();
  void downloadData();
  void unzipData();
  void finishDownload();
  void downloadDataBlocking();

private:
  XnatBrowserWidget* browser;
  XnatDownloadDialog* downloadDialog;

  XnatSettings* settings;

  QString currDir;
  QString zipFilename;
  XnatRestAsynStatus finished;
  unsigned long totalBytes;

  QString xnatFilename;
  QString outFilename;
  QString tempFilePath;
};

#endif
