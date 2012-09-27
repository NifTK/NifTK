#ifndef XnatDownloadManager_h
#define XnatDownloadManager_h

#include "XnatRestWidgetsExports.h"

#include <QObject>

extern "C"
{
#include "XnatRest.h"
}

class QString;
class QWidget;
class XnatDownloadDialog;
class XnatSettings;
class XnatTreeView;

class XnatRestWidgets_EXPORT XnatDownloadManager : public QObject
{
  Q_OBJECT

public:
  XnatDownloadManager(XnatTreeView* xnatTreeView);

  void setSettings(XnatSettings* settings);

  void downloadFile(const QString& fname);
  void silentlyDownloadFile(const QString& fname, const QString& dir);
  void silentlyDownloadAllFiles(const QString& dir);

signals:
  void done();

public slots:
  void downloadFile();
  void downloadAllFiles();

private slots:
  bool startFileDownload(const QString& zipFilename);
  bool startFileGroupDownload(const QString& zipFilename);
  void startDownload();
  void startGroupDownload();
  void downloadData();
  void unzipData();
  void finishDownload();
  void downloadDataBlocking();

private:
  XnatTreeView* xnatTreeView;
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
