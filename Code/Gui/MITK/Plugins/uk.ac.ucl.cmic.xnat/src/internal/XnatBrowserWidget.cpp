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

#include <mitkDataNodeFactory.h>

// Local includes:
#include "XnatSettings.h"
#include "XnatDownloadManager.h"
#include "XnatUploadManager.h"

class XnatBrowserWidgetPrivate
{
public:
  XnatSettings* settings;

  XnatConnection* connection;
  XnatDownloadManager* downloadManager;
  XnatUploadManager* uploadManager;

  QAction* downloadAction;
  QAction* downloadAllAction;
  QAction* importAction;
  QAction* importAllAction;
  QAction* uploadAction;
  QAction* saveDataAndUploadAction;
  QAction* createAction;
  QAction* deleteAction;

  mitk::DataStorage::Pointer dataStorage;
};

XnatBrowserWidget::XnatBrowserWidget(QWidget* parent, Qt::WindowFlags flags)
: QWidget(parent, flags)
, ui(0)
, d_ptr(new XnatBrowserWidgetPrivate())
{
  Q_D(XnatBrowserWidget);

  // initialize data members
  d->settings = 0;
  d->connection = 0;
  d->downloadManager = 0;
  d->uploadManager = new XnatUploadManager(this);

  if (!ui)
  {
    // Create UI
    ui = new Ui::XnatBrowserWidget();
    ui->setupUi(parent);

    ui->middleButtonPanel->setCollapsed(true);

    ui->refreshButton->setEnabled(false);
    ui->downloadButton->setEnabled(false);
    ui->downloadAllButton->setEnabled(false);
    ui->importButton->setEnabled(false);
    ui->importAllButton->setEnabled(false);
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

  if (ui)
  {
    delete ui;
  }
}

XnatSettings* XnatBrowserWidget::settings() const
{
  Q_D(const XnatBrowserWidget);

  return d->settings;
}

void XnatBrowserWidget::setSettings(XnatSettings* settings)
{
  Q_D(XnatBrowserWidget);
  d->settings = settings;
}

mitk::DataStorage::Pointer XnatBrowserWidget::dataStorage() const
{
  Q_D(const XnatBrowserWidget);

  return d->dataStorage;
}

void XnatBrowserWidget::setDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  Q_D(XnatBrowserWidget);
  d->dataStorage = dataStorage;
}

void XnatBrowserWidget::createConnections()
{
  Q_D(XnatBrowserWidget);

  // create actions for popup menus
  d->downloadAction = new QAction(tr("Download"), this);
  connect(d->downloadAction, SIGNAL(triggered()), this, SLOT(downloadFile()));
  d->downloadAllAction = new QAction(tr("Download All"), this);
  connect(d->downloadAllAction, SIGNAL(triggered()), this, SLOT(downloadAllFiles()));
  d->importAction = new QAction(tr("Import"), this);
  connect(d->importAction, SIGNAL(triggered()), this, SLOT(importFile()));
  d->importAllAction = new QAction(tr("Import All"), this);
  connect(d->importAllAction, SIGNAL(triggered()), this, SLOT(importFiles()));
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
  connect(ui->refreshButton, SIGNAL(clicked()), ui->xnatTreeView, SLOT(refreshRows()));
  connect(ui->downloadButton, SIGNAL(clicked()), this, SLOT(downloadFile()));
  connect(ui->downloadAllButton, SIGNAL(clicked()), this, SLOT(downloadAllFiles()));
  connect(ui->importButton, SIGNAL(clicked()), this, SLOT(importFile()));
  connect(ui->importAllButton, SIGNAL(clicked()), this, SLOT(importFiles()));
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

void XnatBrowserWidget::loginXnat()
{
  Q_D(XnatBrowserWidget);

  // show dialog for user to login to XNAT
  XnatConnectDialog* connectDialog = new XnatConnectDialog(XnatConnectionFactory::instance(), this);
  connectDialog->setSettings(d->settings);
  if (connectDialog->exec())
  {
    // delete old connection
    if ( d->connection )
    {
      delete d->connection;
      d->connection = 0;
    }
    // get connection object
    d->connection = connectDialog->getConnection();

    ui->xnatTreeView->initialize(d->connection->getRoot());

    ui->downloadButton->setEnabled(false);
    ui->downloadAllButton->setEnabled(false);
    ui->importButton->setEnabled(false);
    ui->importAllButton->setEnabled(false);
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
//  QModelIndex index = ui->xnatTreeView->selectionModel()->currentIndex();
  QModelIndex index = ui->xnatTreeView->currentIndex();
  XnatModel* model = ui->xnatTreeView->xnatModel();
  QString xnatFilename = model->data(index, Qt::DisplayRole).toString();
  if ( xnatFilename.isEmpty() )
  {
    return;
  }

  // download file
  if ( !d->downloadManager )
  {
    d->downloadManager = new XnatDownloadManager(this);
  }
  QString filename = QFileInfo(xnatFilename).fileName();
  d->downloadManager->downloadFile(filename);
}

void XnatBrowserWidget::importFile()
{
  Q_D(XnatBrowserWidget);

  // get name of file to be downloaded
  QModelIndex index = ui->xnatTreeView->currentIndex();
  XnatModel* model = ui->xnatTreeView->xnatModel();
  QString xnatFilename = model->data(index, Qt::DisplayRole).toString();
  if ( xnatFilename.isEmpty() )
  {
    return;
  }

  // download file
  if ( !d->downloadManager )
  {
    d->downloadManager = new XnatDownloadManager(this);
  }
  QString xnatFileNameTemp = QFileInfo(xnatFilename).fileName();
  QString tempWorkDirectory = d->settings->getWorkSubdirectory();
  d->downloadManager->silentlyDownloadFile(xnatFileNameTemp, tempWorkDirectory);

  // create list of files to open in CAWorks
  QStringList files;
  files.append(QFileInfo(tempWorkDirectory, xnatFileNameTemp).absoluteFilePath());
  if ( files.empty() )
  {
    return;
  }

  // for performance, only check if the first file is readable
  for ( int i = 0 ; i < 1 /*files.size()*/ ; i++ )
  {
    if ( !QFileInfo(files[i]).isReadable() )
    {
      //qWarning() << "File '" << files[i] << "' cannot be read. Type not recognized";
      QString tempString("File '");
      tempString.append(files[i]);
      tempString.append("' cannot be read. Type not recognized");
      QMessageBox::warning(this, tr("Download and Open Error"), tempString);
      return;
    }
  }

  try
  {
    mitk::DataNodeFactory::Pointer nodeFactory = mitk::DataNodeFactory::New();
  //  mitk::FileReader::Pointer fileReader = mitk::FileReader::New();
    // determine reader type based on first file. For now, we are relying
    // on the user to avoid mixing file types.
    QString filename = files[0];
    MITK_INFO << "XnatBrowserWidget::importFile() filename: " << filename.toStdString();
    MITK_INFO << "XnatBrowserWidget::importFile() xnat filename: " << xnatFilename.toStdString();
    nodeFactory->SetFileName(filename.toStdString());
    nodeFactory->Update();
    mitk::DataNode::Pointer dataNode = nodeFactory->GetOutput();
    dataNode->SetName(xnatFilename.toStdString());
    MITK_INFO << "reading the image has succeeded";
    if (d->dataStorage.IsNotNull())
    {
      d->dataStorage->Add(dataNode);
    }
  }
  catch (std::exception& exc)
  {
    MITK_INFO << "reading the image has failed";
  }


}

void XnatBrowserWidget::importFiles()
{
  Q_D(XnatBrowserWidget);

  // get name of file to be downloaded
  QModelIndex index = ui->xnatTreeView->currentIndex();
  XnatModel* model = ui->xnatTreeView->xnatModel();
  QString xnatFilename = model->data(index, Qt::DisplayRole).toString();
  if ( xnatFilename.isEmpty() )
  {
    return;
  }

  // download file
  if ( !d->downloadManager )
  {
    d->downloadManager = new XnatDownloadManager(this);
  }
  QString xnatFileNameTemp = QFileInfo(xnatFilename).fileName();
  QString tempWorkDirectory = d->settings->getWorkSubdirectory();
  d->downloadManager->downloadAllFiles();

//  // create list of files to open in CAWorks
//  QStringList files;
//  files.append(QFileInfo(tempWorkDirectory, xnatFileNameTemp).absoluteFilePath());
//  if ( files.empty() )
//  {
//    return;
//  }
//
//  // for performance, only check if the first file is readable
//  for ( int i = 0 ; i < 1 /*files.size()*/ ; i++ )
//  {
//    if ( !QFileInfo(files[i]).isReadable() )
//    {
//      //qWarning() << "File '" << files[i] << "' cannot be read. Type not recognized";
//      QString tempString("File '");
//      tempString.append(files[i]);
//      tempString.append("' cannot be read. Type not recognized");
//      QMessageBox::warning(this, tr("Download and Open Error"), tempString);
//      return;
//    }
//  }
//
//  try
//  {
//    mitk::DataNodeFactory::Pointer nodeFactory = mitk::DataNodeFactory::New();
//  //  mitk::FileReader::Pointer fileReader = mitk::FileReader::New();
//    // determine reader type based on first file. For now, we are relying
//    // on the user to avoid mixing file types.
//    QString filename = files[0];
//    MITK_INFO << "XnatBrowserWidget::importFile() filename: " << filename.toStdString();
//    MITK_INFO << "XnatBrowserWidget::importFile() xnat filename: " << xnatFilename.toStdString();
//    nodeFactory->SetFileName(filename.toStdString());
//    nodeFactory->Update();
//    mitk::DataNode::Pointer dataNode = nodeFactory->GetOutput();
//    dataNode->SetName(xnatFilename.toStdString());
//    MITK_INFO << "reading the image has succeeded";
//    if (d->dataStorage.IsNotNull())
//    {
//      d->dataStorage->Add(dataNode);
//    }
//  }
//  catch (std::exception& exc)
//  {
//    MITK_INFO << "reading the image has failed";
//  }


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
//  QString groupname = model->name(index);
  QString groupname = model->data(index, Qt::DisplayRole).toString();
  if ( groupname.isEmpty() )
  {
    return;
  }

  // download files
  if ( !d->downloadManager )
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

void XnatBrowserWidget::setButtonEnabled(const QModelIndex& index)
{
  Q_D(XnatBrowserWidget);

  const XnatNodeProperties& nodeProperties = ui->xnatTreeView->nodeProperties(index);
  ui->downloadButton->setEnabled(nodeProperties.isFile());
  ui->downloadAllButton->setEnabled(nodeProperties.holdsFiles());
  ui->importButton->setEnabled(nodeProperties.isFile());
  ui->importAllButton->setEnabled(nodeProperties.holdsFiles());
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
        actions.append(d->importAllAction);
    }
    if ( nodeProperties.isFile() )
    {
      actions.append(d->importAction);
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
