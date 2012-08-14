#ifndef XnatBrowserWidget_h
#define XnatBrowserWidget_h

#include <QWidget>

#include "XnatConnection.h"
#include "XnatDownloadManager.h"
#include "XnatUploadManager.h"
#include "vtkStdString.h"

#include "ui_XnatBrowserWidget.h"

class QPushButton;
class QModelIndex;
class QTreeView;
class QLineEdit;
class QDialog;

class XnatBrowserWidget : public QWidget
{
    Q_OBJECT

    public:
        explicit XnatBrowserWidget(QWidget* parent = 0, Qt::WindowFlags flags = 0);
        virtual ~XnatBrowserWidget();

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

//        QPushButton* loginButton;
//        QPushButton* setDefaultWorkDirectoryButton;
        QPushButton* refreshButton;
//        QPushButton* helpButton;
        QPushButton* downloadButton;
        QPushButton* downloadAllButton;
//        QPushButton* downloadAndOpenButton;
        QPushButton* uploadButton;
//        QPushButton* saveDataAndUploadButton;
        QPushButton* createButton;
        QPushButton* deleteButton;

//        QLineEdit* workDirectoryEdit;
        QDialog* helpDialog;

//        QTreeView* xnatTreeView;

        void initializeTreeView(XnatNode* rootNode);

        /// \brief All the controls for the main view part.
        Ui::XnatBrowserWidget* ui;

        Q_DISABLE_COPY(XnatBrowserWidget);
};

#endif
