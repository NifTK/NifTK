#ifndef XnatUploadManager_h
#define XnatUploadManager_h

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


class XnatUploadManager : public QObject
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
