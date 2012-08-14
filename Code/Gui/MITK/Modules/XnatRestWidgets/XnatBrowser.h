#ifndef XNATBROWSER_H
#define XNATBROWSER_H

#include <QWidget>

#include "XnatConnection.h"
#include "XnatDownloadManager.h"
#include "XnatUploadManager.h"
#include "vtkStdString.h"

class QPushButton;
class QModelIndex;
class QTreeView;
class QLineEdit;
class QDialog;

class XnatBrowser : public QWidget
{
    Q_OBJECT

    public:
        explicit XnatBrowser(QWidget* parent = 0, Qt::WindowFlags flags = 0)
        : QWidget(parent, flags)
        {
          this->constructor();
        }
        virtual ~XnatBrowser();

        bool startFileUpload(const QString& zipFilename);
        bool startFileDownload(const QString& zipFilename);
        bool startFileGroupDownload(const QString& zipFilename);

    public slots:
        void refreshRows();

    private slots:
        void loginXnat();
        void downloadFile();
        void downloadAllFiles();
        void downloadAndOpenFile();
        void createNewRow();
        void deleteRow();
        void setButtonEnabled(const QModelIndex& index);
        void setSaveDataAndUploadButtonEnabled();
        void showContextMenu(const QPoint&);
        void setDefaultWorkDirectory();
        void help();

    private:
        XnatConnection* connection;
        XnatDownloadManager* downloadManager;
        XnatUploadManager* uploadManager;

        QAction* downloadAction;
        QAction* downloadAllAction;
        QAction* downloadAndOpenAction;
        QAction* uploadAction;
        QAction* saveDataAndUploadAction;
        QAction* createAction;
        QAction* deleteAction;

        QPushButton* loginButton;
        QPushButton* setDefaultWorkDirectoryButton;
        QPushButton* refreshButton;
        QPushButton* helpButton;
        QPushButton* downloadButton;
        QPushButton* downloadAllButton;
        QPushButton* downloadAndOpenButton;
        QPushButton* uploadButton;
        QPushButton* saveDataAndUploadButton;
        QPushButton* createButton;
        QPushButton* deleteButton;

        QLineEdit* workDirectoryEdit;
        QDialog* helpDialog;

        QTreeView* xnatTreeView;

        void constructor();
        void initializeTreeView(XnatNode* rootNode);

        Q_DISABLE_COPY(XnatBrowser);
};

#endif
