#include "XnatBrowserWidget.h"

// XnatRestWidgets module includes
#include <XnatConnection.h>
#include <XnatConnectDialog.h>
#include <XnatException.h>
#include <XnatModel.h>
#include <XnatNameDialog.h>
#include <XnatNodeActivity.h>
#include <XnatNodeProperties.h>
#include <XnatTreeView.h>

// VTK includes
//#include <vtkStdString.h>

// Qt includes
#include <QAction>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QMenu>
#include <QTextBrowser>

#include <QDebug>

// Local includes:
#include "XnatBrowserSettings.h"
#include "XnatDownloadManager.h"
#include "XnatUploadManager.h"

class XnatBrowserWidgetPrivate
{
public:
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

  QDialog* helpDialog;
};

XnatBrowserWidget::XnatBrowserWidget(QWidget* parent, Qt::WindowFlags flags)
: QWidget(parent, flags)
, ui(0)
, d_ptr(new XnatBrowserWidgetPrivate())
{
  Q_D(XnatBrowserWidget);

  // initialize data members
  d->connection = 0;
  d->downloadManager = 0;
  d->uploadManager = new XnatUploadManager(this);
  d->helpDialog = 0;

  if (!ui)
  {
    // Create UI
    ui = new Ui::XnatBrowserWidget();
    ui->setupUi(parent);

    ui->middleButtonPanel->setCollapsed(true);

    // create line edit widget to display work directory
    ui->workDirectoryEdit->setText(QDir::toNativeSeparators(XnatBrowserSettings::getDefaultWorkDirectory()));
    ui->workDirectoryEdit->setReadOnly(true);

    ui->refreshButton->setEnabled(false);
    ui->downloadButton->setEnabled(false);
    ui->downloadAllButton->setEnabled(false);
    ui->downloadAndOpenButton->setEnabled(false);
    ui->uploadButton->setEnabled(false);
    ui->saveDataAndUploadButton->setEnabled(false);
    ui->createButton->setEnabled(false);
    ui->deleteButton->setEnabled(false);

    // Create connections after setting defaults, so you don't trigger stuff when setting defaults.
    createConnections();
  }
}

XnatBrowserWidget::~XnatBrowserWidget()
{
  Q_D(XnatBrowserWidget);

  // clean up XNAT connection
  if (d->connection)
  {
    delete d->connection;
    d->connection = 0;
  }

  // clean up download manager
  if (d->downloadManager)
  {
    delete d->downloadManager;
    d->downloadManager = 0;
  }

  // clean up upload manager
  delete d->uploadManager;
  d->uploadManager = 0;

  // delete help browser dialog
  if (d->helpDialog)
  {
    delete d->helpDialog;
    d->helpDialog = 0;
  }

  if (ui)
  {
    delete ui;
  }
}

void XnatBrowserWidget::createConnections()
{
  Q_D(XnatBrowserWidget);

  // create actions for popup menus
  d->downloadAction = new QAction(tr("Download"), this);
  connect(d->downloadAction, SIGNAL(triggered()), this, SLOT(downloadFile()));
  d->downloadAllAction = new QAction(tr("Download All"), this);
  connect(d->downloadAllAction, SIGNAL(triggered()), this, SLOT(downloadAllFiles()));
  d->downloadAndOpenAction = new QAction(tr("Download And Open"), this);
  connect(d->downloadAndOpenAction, SIGNAL(triggered()), this, SLOT(downloadAndOpenFile()));
  d->uploadAction = new QAction(tr("Upload"), this);
  connect(d->uploadAction, SIGNAL(triggered()), d->uploadManager, SLOT(uploadFiles()));
  d->saveDataAndUploadAction = new QAction(tr("Save Data and Upload"), this);
//    new XnatReactionSaveData(saveDataAndUploadAction, uploadManager, this);
  d->createAction = new QAction(tr("Create New"), this);
  connect(d->createAction, SIGNAL(triggered()), this, SLOT(createNewRow()));
  d->deleteAction = new QAction(tr("Delete"), this);
  connect(d->deleteAction, SIGNAL(triggered()), this, SLOT(deleteCurrentRow()));

  // create button widgets
  connect(ui->loginButton, SIGNAL(clicked()), this, SLOT(loginXnat()));
  connect(ui->setDefaultWorkDirectoryButton, SIGNAL(clicked()), this, SLOT(setDefaultWorkDirectory()));
  connect(ui->refreshButton, SIGNAL(clicked()), ui->xnatTreeView, SLOT(refreshRows()));
  connect(ui->helpButton, SIGNAL(clicked()), this, SLOT(help()));
  connect(ui->downloadButton, SIGNAL(clicked()), this, SLOT(downloadFile()));
  connect(ui->downloadAllButton, SIGNAL(clicked()), this, SLOT(downloadAllFiles()));
  connect(ui->downloadAndOpenButton, SIGNAL(clicked()), this, SLOT(downloadAndOpenFile()));
  connect(ui->uploadButton, SIGNAL(clicked()), d->uploadManager, SLOT(uploadFiles()));
  connect(ui->saveDataAndUploadButton, SIGNAL(clicked()), d->saveDataAndUploadAction, SLOT(trigger()));
  connect(d->saveDataAndUploadAction, SIGNAL(changed()), this, SLOT(setSaveDataAndUploadButtonEnabled()));
  connect(ui->createButton, SIGNAL(clicked()), this, SLOT(createNewRow()));
  connect(ui->deleteButton, SIGNAL(clicked()), this, SLOT(deleteCurrentRow()));

  connect(ui->xnatTreeView, SIGNAL(clicked(const QModelIndex&)), this, SLOT(setButtonEnabled(const QModelIndex&)));

  ui->xnatTreeView->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(ui->xnatTreeView, SIGNAL(customContextMenuRequested(const QPoint&)),
          this, SLOT(showContextMenu(const QPoint&)));
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

    ui->workDirectoryEdit->setText(QDir::toNativeSeparators(dirPath));

    XnatBrowserSettings::setDefaultWorkDirectory(dirPath);
  }
}

void XnatBrowserWidget::loginXnat()
{
  Q_D(XnatBrowserWidget);

  // show dialog for user to login to XNAT
  XnatConnectDialog* connectDialog = new XnatConnectDialog(XnatConnectionFactory::instance(), this);
  if (connectDialog->exec())
  {
    // delete old connection
    if ( d->connection != 0 )
    {
      delete d->connection;
      d->connection = 0;
    }
    // get connection object
    d->connection = connectDialog->getConnection();

    ui->xnatTreeView->initialize(d->connection->getRoot());

    ui->downloadButton->setEnabled(false);
    ui->downloadAllButton->setEnabled(false);
    ui->downloadAndOpenButton->setEnabled(false);
    ui->uploadButton->setEnabled(false);
    ui->saveDataAndUploadButton->setEnabled(false);
    ui->createButton->setEnabled(false);
    ui->deleteButton->setEnabled(false);

    ui->refreshButton->setEnabled(true);
  }
  delete connectDialog;
}

void XnatBrowserWidget::refreshRows()
{
  ui->xnatTreeView->refreshRows();
}

void XnatBrowserWidget::downloadFile()
{
  Q_D(XnatBrowserWidget);

  // get name of file to be downloaded
  QModelIndex index = ui->xnatTreeView->selectionModel()->currentIndex();
  XnatModel* model = ui->xnatTreeView->xnatModel();
  QString xnatFilename = model->name(index);
  if ( xnatFilename.isEmpty() )
  {
    return;
  }

  // download file
  if ( d->downloadManager )
  {
    d->downloadManager = new XnatDownloadManager(this);
  }
  d->downloadManager->downloadFile(QFileInfo(xnatFilename).fileName());
}

void XnatBrowserWidget::downloadAndOpenFile()
{
  Q_D(XnatBrowserWidget);

  // get name of file to be downloaded
  QModelIndex index = ui->xnatTreeView->selectionModel()->currentIndex();
  XnatModel* model = ui->xnatTreeView->xnatModel();
  QString xnatFilename = model->name(index);
  if ( xnatFilename.isEmpty() )
  {
    return;
  }

  // download file
  if ( d->downloadManager )
  {
    d->downloadManager = new XnatDownloadManager(this);
  }
  QString xnatFileNameTemp = QFileInfo(xnatFilename).fileName();
  QString tempWorkDirectory = XnatBrowserSettings::getWorkSubdirectory();
  d->downloadManager->silentlyDownloadFile(xnatFileNameTemp, tempWorkDirectory);

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
    QModelIndex index = ui->xnatTreeView->selectionModel()->currentIndex();
    XnatModel* model = ui->xnatTreeView->xnatModel();
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
  Q_D(XnatBrowserWidget);

  // get name of file group to be downloaded
  QModelIndex index = ui->xnatTreeView->selectionModel()->currentIndex();
  XnatModel* model = ui->xnatTreeView->xnatModel();
  QString groupname = model->name(index);
  if ( groupname.isEmpty() )
  {
    return;
  }

  // download files
  if ( d->downloadManager )
  {
      d->downloadManager = new XnatDownloadManager(this);
  }
  d->downloadManager->downloadAllFiles();
}

bool XnatBrowserWidget::startFileGroupDownload(const QString& zipFilename)
{
  // start download of zip file containing selected file group
  try
  {
    QModelIndex index = ui->xnatTreeView->selectionModel()->currentIndex();
    ui->xnatTreeView->xnatModel()->downloadFileGroup(index, zipFilename);
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
    QModelIndex index = ui->xnatTreeView->selectionModel()->currentIndex();
    XnatModel* model = ui->xnatTreeView->xnatModel();
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
  ui->xnatTreeView->createNewRow();
}

void XnatBrowserWidget::deleteCurrentRow()
{
  ui->xnatTreeView->deleteCurrentRow();
}

void XnatBrowserWidget::help()
{
  Q_D(XnatBrowserWidget);

  if ( !d->helpDialog )
  {
    QTextBrowser* helpBrowser = new QTextBrowser;
    helpBrowser->setSearchPaths(QStringList() << ":/XnatHelp");
    helpBrowser->setSource(QString("index.html"));
    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(helpBrowser);

    d->helpDialog = new QDialog(this);
    d->helpDialog->setLayout(layout);
    d->helpDialog->setWindowTitle(tr("XNAT Browser Help"));
    d->helpDialog->resize(700, 480);
  }

  d->helpDialog->show();
}

void XnatBrowserWidget::setButtonEnabled(const QModelIndex& index)
{
  Q_D(XnatBrowserWidget);

  const XnatNodeProperties& nodeProperties = ui->xnatTreeView->nodeProperties(index);
  ui->downloadButton->setEnabled(nodeProperties.isFile());
  ui->downloadAllButton->setEnabled(nodeProperties.holdsFiles());
  ui->downloadAndOpenButton->setEnabled(nodeProperties.isFile());
  ui->uploadButton->setEnabled(nodeProperties.receivesFiles());
  ui->saveDataAndUploadButton->setEnabled((nodeProperties.receivesFiles() && d->saveDataAndUploadAction->isEnabled()));
  ui->createButton->setEnabled(nodeProperties.isModifiable());
  ui->deleteButton->setEnabled(nodeProperties.isDeletable());
}

void XnatBrowserWidget::setSaveDataAndUploadButtonEnabled()
{
  Q_D(XnatBrowserWidget);

  const XnatNodeProperties& nodeProperties = ui->xnatTreeView->currentNodeProperties();
  ui->saveDataAndUploadButton->setEnabled((nodeProperties.receivesFiles() && d->saveDataAndUploadAction->isEnabled()));
}

void XnatBrowserWidget::showContextMenu(const QPoint& position)
{
  Q_D(XnatBrowserWidget);

  const QModelIndex& index = ui->xnatTreeView->indexAt(position);
  if ( index.isValid() )
  {
    XnatNodeProperties nodeProperties = ui->xnatTreeView->nodeProperties(index);
    QList<QAction*> actions;
    if ( nodeProperties.isFile() )
    {
      actions.append(d->downloadAction);
    }
    if ( nodeProperties.holdsFiles() )
    {
      actions.append(d->downloadAllAction);
    }
    if ( nodeProperties.isFile() )
    {
      actions.append(d->downloadAndOpenAction);
    }
    if ( nodeProperties.receivesFiles() )
    {
      actions.append(d->uploadAction);

      if ( d->saveDataAndUploadAction->isEnabled() )
      {
        actions.append(d->saveDataAndUploadAction);
      }
    }
    if ( nodeProperties.isModifiable() )
    {
      actions.append(d->createAction);
    }
    if ( nodeProperties.isDeletable() )
    {
      actions.append(d->deleteAction);
    }
    if ( actions.count() > 0 )
    {
      QMenu::exec(actions, ui->xnatTreeView->mapToGlobal(position));
    }
  }
}
