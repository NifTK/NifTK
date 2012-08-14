#ifndef XNATUPLOADMANAGER_H
#define XNATUPLOADMANAGER_H

#include <QObject>
extern "C" {
#include "XnatRest.h"
}
#include "XnatUploadDialog.h"

class QString;
class QStringList;
class XnatBrowser;


class XnatUploadManager : public QObject
{
    Q_OBJECT

    public:
        XnatUploadManager(XnatBrowser* b);
        void uploadSavedData(const QString& dir);

    public slots:
        void uploadFiles();

    private slots:
        void zipFiles();
        void startUpload();
        void uploadData();

    private:
        XnatBrowser* browser;
        XnatUploadDialog* uploadDialog;

        QString currDir;
        QStringList userFilePaths;
        QString zipFilename;
        XnatRestAsynStatus finished;
        unsigned long totalBytes;

        bool getFilenames();
};

#endif
