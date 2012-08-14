#ifndef XNATDOWNLOADMANAGER_H
#define XNATDOWNLOADMANAGER_H

#include <QObject>
extern "C" {
#include "XnatRest.h"
}
#include "XnatDownloadDialog.h"

class QString;
class XnatBrowser;


class XnatDownloadManager : public QObject
{
    Q_OBJECT

    public:
        XnatDownloadManager(XnatBrowser* b);
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
        XnatBrowser* browser;
        XnatDownloadDialog* downloadDialog;

        QString currDir;
        QString zipFilename;
        XnatRestAsynStatus finished;
        unsigned long totalBytes;

        QString xnatFilename;
        QString outFilename;
        QString tempFilePath;
};

#endif
