#include <QtGui>
extern "C" {
#include "XnatRest.h"
}
#include "XnatBrowserSettings.h"
#include "XnatConnectDialog.h"
#include "XnatModel.h"
#include "XnatNodeActivity.h"
#include "XnatNodeProperties.h"
#include "XnatNameDialog.h"
#include "XnatException.h"
#include "XnatBrowserWidget.h"

#include <QDir>

void XnatBrowserWidget::constructor()
{
    // initialize data members
    connection = NULL;
    downloadManager = NULL;
    uploadManager = new XnatUploadManager(this);
    helpDialog = NULL;

    // create actions for popup menus
    downloadAction = new QAction(tr("Download"), this);
    connect(downloadAction, SIGNAL(triggered()), this, SLOT(downloadFile()));
    downloadAllAction = new QAction(tr("Download All"), this);
    connect(downloadAllAction, SIGNAL(triggered()), this, SLOT(downloadAllFiles()));
    downloadAndOpenAction = new QAction(tr("Download And Open"), this);
    connect(downloadAndOpenAction, SIGNAL(triggered()), this, SLOT(downloadAndOpenFile()));
    uploadAction = new QAction(tr("Upload"), this);
    connect(uploadAction, SIGNAL(triggered()), uploadManager, SLOT(uploadFiles()));
    saveDataAndUploadAction = new QAction(tr("Save Data and Upload"), this);
//    new XnatReactionSaveData(saveDataAndUploadAction, uploadManager, this);
    createAction = new QAction(tr("Create New"), this);
    connect(createAction, SIGNAL(triggered()), this, SLOT(createNewRow()));
    deleteAction = new QAction(tr("Delete"), this);
    connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteRow()));

    // create button widgets
    loginButton = new QPushButton(tr("Login"));
    connect(loginButton, SIGNAL(clicked()), this, SLOT(loginXnat()));
    setDefaultWorkDirectoryButton = new QPushButton(tr("Set Work Directory"));
    connect(setDefaultWorkDirectoryButton, SIGNAL(clicked()), this, SLOT(setDefaultWorkDirectory()));
    refreshButton = new QPushButton(tr("Refresh"));
    refreshButton->setEnabled(false);
    connect(refreshButton, SIGNAL(clicked()), this, SLOT(refreshRows()));
    helpButton = new QPushButton(tr("Help"));
    connect(helpButton, SIGNAL(clicked()), this, SLOT(help()));
    downloadButton = new QPushButton(tr("Download"));
    connect(downloadButton, SIGNAL(clicked()), this, SLOT(downloadFile()));
    downloadAllButton = new QPushButton(tr("Download All"));
    connect(downloadAllButton, SIGNAL(clicked()), this, SLOT(downloadAllFiles()));
    downloadAndOpenButton = new QPushButton(tr("Download and Open"));
    connect(downloadAndOpenButton, SIGNAL(clicked()), this, SLOT(downloadAndOpenFile()));
    uploadButton = new QPushButton(tr("Upload"));
    connect(uploadButton, SIGNAL(clicked()), uploadManager, SLOT(uploadFiles()));
    saveDataAndUploadButton = new QPushButton(tr("Save Data and Upload"));
    connect(saveDataAndUploadButton, SIGNAL(clicked()), saveDataAndUploadAction, SLOT(trigger()));
    connect(saveDataAndUploadAction, SIGNAL(changed()), this, SLOT(setSaveDataAndUploadButtonEnabled()));
    createButton = new QPushButton(tr("Create New"));
    connect(createButton, SIGNAL(clicked()), this, SLOT(createNewRow()));
    deleteButton = new QPushButton(tr("Delete"));
    connect(deleteButton, SIGNAL(clicked()), this, SLOT(deleteRow()));

    // create line edit widget to display work directory
    workDirectoryEdit = new QLineEdit(QDir::toNativeSeparators(XnatBrowserSettings::getDefaultWorkDirectory()));
    workDirectoryEdit->setReadOnly(true);

    // create XNAT tree view
    xnatTreeView = new QTreeView;
    xnatTreeView->setSelectionBehavior(QTreeView::SelectItems);
    xnatTreeView->setUniformRowHeights(true);
    xnatTreeView->setHeaderHidden(true);
    initializeTreeView(new XnatNode(XnatEmptyNodeActivity::instance()));
    connect(xnatTreeView, SIGNAL(clicked(QModelIndex)), this, SLOT(setButtonEnabled(QModelIndex)));
    xnatTreeView->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(xnatTreeView, SIGNAL(customContextMenuRequested(const QPoint&)), 
            this, SLOT(showContextMenu(const QPoint&)));

    // create XNAT browser widget
    QGridLayout* topButtonLayout = new QGridLayout;
    topButtonLayout->addWidget(loginButton, 0, 0);
    topButtonLayout->addWidget(setDefaultWorkDirectoryButton, 0, 1);
    topButtonLayout->addWidget(workDirectoryEdit, 0, 2);
    topButtonLayout->addWidget(helpButton, 1, 0);
    topButtonLayout->addWidget(downloadAndOpenButton, 1, 1);
    topButtonLayout->addWidget(saveDataAndUploadButton, 1, 2);
    QGroupBox* topButtonPanel = new QGroupBox;
    topButtonPanel->setLayout(topButtonLayout);

    QGridLayout* middleButtonLayout = new QGridLayout;
    middleButtonLayout->addWidget(refreshButton, 0, 0);
    middleButtonLayout->addWidget(createButton, 0, 1);
    middleButtonLayout->addWidget(deleteButton, 0, 2);
    middleButtonLayout->addWidget(downloadButton, 1, 0);
    middleButtonLayout->addWidget(downloadAllButton, 1, 1);
    middleButtonLayout->addWidget(uploadButton, 1, 2);
//    pqCollapsedGroup* middleButtonPanel = new pqCollapsedGroup;
//    middleButtonPanel->setTitle(tr("More XNAT Functions"));
//    middleButtonPanel->setCollapsed(true);
//    middleButtonPanel->setLayout(middleButtonLayout);

    QVBoxLayout* browserLayout = new QVBoxLayout;
    browserLayout->addWidget(topButtonPanel);
//    browserLayout->addWidget(middleButtonPanel);
    browserLayout->addWidget(xnatTreeView);
    QWidget* browserWidget = new QWidget;
    browserWidget->setLayout(browserLayout);

    // initialize dock widget and set browser widget
    setWindowTitle(tr("XNAT Browser"));
    setObjectName(tr("XnatBrowserWidget"));
//    setAllowedAreas(Qt::RightDockWidgetArea);
//    setWidget(browserWidget);
    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(browserWidget);
    setLayout(layout);
}

XnatBrowserWidget::~XnatBrowserWidget()
{
    // clean up models in tree view
    XnatModel* oldModel = (XnatModel*) xnatTreeView->model();
    QItemSelectionModel* oldSelectionModel = xnatTreeView->selectionModel();
    if ( oldModel != NULL )
    {
        delete oldModel;
    }
    if ( oldSelectionModel != NULL )
    {
        delete oldSelectionModel;
    }

    // clean up XNAT connection
    if ( connection != NULL )
    {
        delete connection;
        connection = NULL;
    }

    // clean up download manager
    if ( downloadManager != NULL )
    {
        delete downloadManager;
        downloadManager = NULL;
    }

    // clean up upload manager
    delete uploadManager;
    uploadManager = NULL;

    // delete help browser dialog
    if ( helpDialog != NULL )
    {
        delete helpDialog;
        helpDialog = NULL;
    }
}

void XnatBrowserWidget::initializeTreeView(XnatNode* rootNode)
{
    XnatModel* oldModel = (XnatModel*) xnatTreeView->model();
    QItemSelectionModel* oldSelectionModel = xnatTreeView->selectionModel();
    xnatTreeView->setModel(new XnatModel(rootNode));
    if ( oldModel != NULL )
    {
        delete oldModel;
    }
    if ( oldSelectionModel != NULL )
    {
        delete oldSelectionModel;
    }
    xnatTreeView->setExpanded(QModelIndex(), false);
    downloadButton->setEnabled(false);
    downloadAllButton->setEnabled(false);
    downloadAndOpenButton->setEnabled(false);
    uploadButton->setEnabled(false);
    saveDataAndUploadButton->setEnabled(false);
    createButton->setEnabled(false);
    deleteButton->setEnabled(false);
}

void XnatBrowserWidget::setDefaultWorkDirectory()
{
    QFileDialog fileDialog(this, QString("Set Work Directory"),
                            XnatBrowserSettings::getDefaultWorkDirectory());

    fileDialog.setFileMode(QFileDialog::Directory);
    // fileDialog.setOption(pqFileDialog::ShowDirsOnly, true);

    if( fileDialog.exec() == QDialog::Accepted )
    {
        QString dirPath = fileDialog.selectedFiles()[0];

        workDirectoryEdit->setText(QDir::toNativeSeparators(dirPath));

        XnatBrowserSettings::setDefaultWorkDirectory(dirPath);
    }
}

void XnatBrowserWidget::loginXnat()
{
    // show dialog for user to login to XNAT
    XnatConnectDialog* connectDialog = new XnatConnectDialog(XnatConnectionFactory::instance(), this);
    if (connectDialog->exec())
    {
        // delete old connection
        if ( connection != NULL )
        {
            delete connection;
            connection = NULL;
        }
        // get connection object
        connection = connectDialog->getConnection();
        // initialize tree view
        initializeTreeView(connection->getRoot());
        refreshButton->setEnabled(true);
    }
    delete connectDialog;
}

void XnatBrowserWidget::refreshRows()
{
    QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
    XnatModel* model = (XnatModel*) xnatTreeView->model();
    model->removeAllRows(index);
    model->fetchMore(index);
}

void XnatBrowserWidget::downloadFile()
{
    // get name of file to be downloaded
    QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
    XnatModel* model = (XnatModel*) xnatTreeView->model();
    QString xnatFilename = model->data(index, Qt::DisplayRole).toString();
    if ( xnatFilename.isEmpty() )
    {
        return;
    }

    // download file
    if ( downloadManager == NULL )
    {
        downloadManager = new XnatDownloadManager(this);
    }
    downloadManager->downloadFile(QFileInfo(xnatFilename).fileName());
}

void XnatBrowserWidget::downloadAndOpenFile()
{
    // get name of file to be downloaded
    QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
    XnatModel* model = (XnatModel*) xnatTreeView->model();
    QString xnatFilename = model->data(index, Qt::DisplayRole).toString();
    if ( xnatFilename.isEmpty() )
    {
        return;
    }

    // download file
    if ( downloadManager == NULL )
    {
        downloadManager = new XnatDownloadManager(this);
    }
    QString xnatFileNameTemp = QFileInfo(xnatFilename).fileName();
    QString tempWorkDirectory = XnatBrowserSettings::getWorkSubdirectory();
    downloadManager->silentlyDownloadFile(xnatFileNameTemp, tempWorkDirectory);

    // create list of files to open in CAWorks
    QStringList files;
    files.append(QFileInfo(tempWorkDirectory, xnatFileNameTemp).absoluteFilePath());
    if ( files.empty() )
    {
        return;
    }

//    pqServer* server = pqActiveObjects::instance().activeServer();
//    if (!server)
//    {
//        //qCritical() << "Cannot create reader without an active server.";
//        return;
//    }
//
//    vtkSMReaderFactory* readerFactory =
//        vtkSMProxyManager::GetProxyManager()->GetReaderFactory();
//
//    // for performance, only check if the first file is readable
//    for ( int i = 0 ; i < 1 /*files.size()*/ ; i++ )
//    {
//        if ( !readerFactory->TestFileReadability(files[i].toAscii().data(), server->GetConnectionID()) )
//        {
//            //qWarning() << "File '" << files[i] << "' cannot be read. Type not recognized";
//            QString tempString("File '");
//            tempString.append(files[i]);
//            tempString.append("' cannot be read. Type not recognized");
//            QMessageBox::warning(this, tr("Download and Open Error"), tempString);
//            return;
//        }
//    }
//
//    // determine reader type based on first file. For now, we are relying
//    // on the user to avoid mixing file types.
//    QString filename = files[0];
//    QString readerType, readerGroup;
//
//    if ( readerFactory->CanReadFile(filename.toAscii().data(), server->GetConnectionID()) )
//    {
//        readerType = readerFactory->GetReaderName();
//        readerGroup = readerFactory->GetReaderGroup();
//    }
//    else
//    {
//        // reader factory could not determine the type of reader to create for the
//        // file. Ask the user.
//        pqSelectReaderDialog prompt(filename, server,
//            readerFactory, pqCoreUtilities::mainWidget());
//        if ( prompt.exec() == QDialog::Accepted )
//        {
//            readerType = prompt.getReader();
//            readerGroup = prompt.getGroup();
//        }
//        else
//        {
//            // user didn't choose any reader
//            return;
//        }
//    }
//
//    BEGIN_UNDO_SET("Create 'Reader'");
//    pqObjectBuilder* builder =
//        pqApplicationCore::instance()->getObjectBuilder();
//    pqPipelineSource* reader = builder->createReader(readerGroup,
//        readerType, files, server);
//
//    if ( reader )
//    {
//        pqApplicationCore* core = pqApplicationCore::instance();
//
//        // Add this to the list of recent server resources ...
//        pqServerResource resource = server->getResource();
//        resource.setPath(files[0]);
//        resource.addData("readergroup", reader->getProxy()->GetXMLGroup());
//        resource.addData("reader", reader->getProxy()->GetXMLName());
//        resource.addData("extrafilesCount", QString("%1").arg(files.size()-1));
//        for ( int cc = 1 ; cc < files.size() ; cc++ )
//        {
//            resource.addData(QString("file.%1").arg(cc-1), files[cc]);
//        }
//        core->serverResources().add(resource);
//        core->serverResources().save(*core->settings());
//    }
//    END_UNDO_SET();
}

bool XnatBrowserWidget::startFileDownload(const QString& zipFilename)
{
    // start download of zip file
    try
    {
        QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
        XnatModel* model = (XnatModel*) xnatTreeView->model();
        model->downloadFile(index, zipFilename);
    }
    catch (XnatException& e)
    {
        QMessageBox::warning(this, tr("Downloaded File Error"), tr(e.what()));
        return false;
    }

    return true;
}

void XnatBrowserWidget::downloadAllFiles()
{
    // get name of file group to be downloaded
    QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
    XnatModel* model = (XnatModel*) xnatTreeView->model();
    QString groupname = model->data(index, Qt::DisplayRole).toString();
    if ( groupname.isEmpty() )
    {
        return;
    }

    // download files
    if ( downloadManager == NULL )
    {
        downloadManager = new XnatDownloadManager(this);
    }
    downloadManager->downloadAllFiles();
}

bool XnatBrowserWidget::startFileGroupDownload(const QString& zipFilename)
{
    // start download of zip file containing selected file group
    try
    {
        QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
        XnatModel* model = (XnatModel*) xnatTreeView->model();
        model->downloadFileGroup(index, zipFilename);
    }
    catch (XnatException& e)
    {
        QMessageBox::warning(this, tr("Download File Group Error"), tr(e.what()));
        return false;
    }

    return true;
}

bool XnatBrowserWidget::startFileUpload(const QString& zipFilename)
{
    // start upload of zip file
    try
    {
        QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
        XnatModel* model = (XnatModel*) xnatTreeView->model();
        model->uploadFile(index, zipFilename);
    }
    catch (XnatException& e)
    {
        QMessageBox::warning(this, tr("Upload Files Error"), tr(e.what()));
        return false;
    }

    return true;
}

void XnatBrowserWidget::createNewRow()
{
    QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
    XnatModel* model = (XnatModel*) xnatTreeView->model();

    // get kind of new entry, e.g., reconstruction or resource
    QString childKind = model->data(index, XnatModel::ModifiableChildKind).toString();
    if ( childKind.isEmpty() )
    {
        QMessageBox::warning(this, tr("Create New Error"), tr("Unknown child kind"));
        return;
    }

    // get parent name, e.g., experiment name for new reconstruction, or 
    //                        reconstruction name for new resource
    QString parentName = model->data(index, XnatModel::ModifiableParentName).toString();
    if ( parentName.isEmpty() )
    {
        QMessageBox::warning(this, tr("Create New Error"), tr("Unknown parent name"));
        return;
    }

    // get name of new child in parent from user, e.g.,
    //             name of new reconstruction in experiment, or
    //             name of new resource in reconstruction
    XnatNameDialog nameDialog(this, childKind, parentName);
    if ( nameDialog.exec() )
    {
        QString name = nameDialog.getNewName();

        try
        {
            // create new child in parent, e.g., new reconstruction in experiment, or
            //                                   new resource in reconstruction
            model->addEntry(index, name);

            // refresh display
            model->removeAllRows(index);
            model->fetchMore(index);
        }
        catch (XnatException& e)
        {
            QMessageBox::warning(this, tr("Create New Error"), tr(e.what()));
        }
    }
}

void XnatBrowserWidget::deleteRow()
{
    // get name in row to be deleted
    QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
    XnatModel* model = (XnatModel*) xnatTreeView->model();
    QString name = model->data(index, Qt::DisplayRole).toString();

    // ask user to confirm deletion
    int buttonPressed = QMessageBox::question(this, tr("Confirm Deletion"), tr("Delete %1 ?").arg(name),
                                              QMessageBox::Yes | QMessageBox::No);

    if ( buttonPressed == QMessageBox::Yes )
    {
        try
        {
            // delete row
            QModelIndex parent = model->parent(index);
            model->removeEntry(index);

            // refresh display
            model->removeAllRows(parent);
            model->fetchMore(parent);
        }
        catch (XnatException& e)
        {
            QMessageBox::warning(this, tr("Delete Error"), tr(e.what()));
        }
    }
}

void XnatBrowserWidget::help()
{
    if ( !helpDialog )
    {
        QTextBrowser* helpBrowser = new QTextBrowser;
        helpBrowser->setSearchPaths(QStringList() << ":/XnatHelp");
        helpBrowser->setSource(QString("index.html"));
        QVBoxLayout* layout = new QVBoxLayout;
        layout->addWidget(helpBrowser);

        helpDialog = new QDialog(this);
        helpDialog->setLayout(layout);
        helpDialog->setWindowTitle(tr("XNAT Browser Help"));
        helpDialog->resize(700, 480);
    }

    helpDialog->show();
}

void XnatBrowserWidget::setButtonEnabled(const QModelIndex& index)
{
    XnatNodeProperties nodeProperties(xnatTreeView->model()->data(index, Qt::UserRole).toBitArray());
    downloadButton->setEnabled(nodeProperties.isFile());
    downloadAllButton->setEnabled(nodeProperties.holdsFiles());
    downloadAndOpenButton->setEnabled(nodeProperties.isFile());
    uploadButton->setEnabled(nodeProperties.receivesFiles());
    saveDataAndUploadButton->setEnabled((nodeProperties.receivesFiles() && saveDataAndUploadAction->isEnabled()));
    createButton->setEnabled(nodeProperties.isModifiable());
    deleteButton->setEnabled(nodeProperties.isDeletable());
}

void XnatBrowserWidget::setSaveDataAndUploadButtonEnabled()
{
    QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
    XnatNodeProperties nodeProperties(xnatTreeView->model()->data(index, Qt::UserRole).toBitArray());
    saveDataAndUploadButton->setEnabled((nodeProperties.receivesFiles() && saveDataAndUploadAction->isEnabled()));
}

void XnatBrowserWidget::showContextMenu(const QPoint& position)
{
    QModelIndex index = xnatTreeView->indexAt(position);
    if ( index.isValid() )
    {
        XnatNodeProperties nodeProperties(xnatTreeView->model()->data(index, Qt::UserRole).toBitArray());
        QList<QAction*> actions;
        if ( nodeProperties.isFile() )
        {
            actions.append(downloadAction);
        }
        if ( nodeProperties.holdsFiles() )
        {
            actions.append(downloadAllAction);
        }
        if ( nodeProperties.isFile() )
        {
            actions.append(downloadAndOpenAction);
        }
        if ( nodeProperties.receivesFiles() )
        {
            actions.append(uploadAction);

            if ( saveDataAndUploadAction->isEnabled() )
            {
                actions.append(saveDataAndUploadAction);
            }
        }
        if ( nodeProperties.isModifiable() )
        {
            actions.append(createAction);
        }
        if ( nodeProperties.isDeletable() )
        {
            actions.append(deleteAction);
        }
        if ( actions.count() > 0 )
        {
            QMenu::exec(actions, xnatTreeView->mapToGlobal(position));
        }
    }
}
