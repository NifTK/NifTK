#ifndef XnatUploadManager_h
#define XnatUploadManager_h

#include <QObject>
#include <QString>
#include <QStringList>

extern "C"
{
#include "XnatRest.h"
}

class XnatBrowserWidget;
class XnatSettings;
class XnatUploadDialog;


class XnatUploadManager : public QObject
{
  Q_OBJECT

public:
  XnatUploadManager(XnatBrowserWidget* b);
  void uploadSavedData(const QString& dir);

public slots:
  void uploadFiles();

private slots:
  void zipFiles();
  void startUpload();
  void uploadData();

private:
  bool getFilenames();

  XnatBrowserWidget* browser;
  XnatUploadDialog* uploadDialog;

  QString currDir;
  QStringList userFilePaths;
  QString zipFilename;
  XnatRestAsynStatus finished;
  unsigned long totalBytes;

  XnatSettings* settings;
};

#endif
