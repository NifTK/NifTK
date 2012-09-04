#ifndef XnatDownloadManager_h
#define XnatDownloadManager_h

#include <QObject>

extern "C"
{
#include "XnatRest.h"
}

class QString;
class QWidget;
class XnatBrowserWidget;
class XnatDownloadDialog;
class XnatSettings;

class XnatDownloadManager : public QObject
{
  Q_OBJECT

public:
  XnatDownloadManager(XnatBrowserWidget* b, XnatSettings* settings);
  void downloadFile(const QString& fname);
  void downloadAllFiles();
  void silentlyDownloadFile(const QString& fname, const QString& dir);
  void silentlyDownloadAllFiles(const QString& dir);

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
  QWidget* parent;
//  XnatBrowserWidget* browser;
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
