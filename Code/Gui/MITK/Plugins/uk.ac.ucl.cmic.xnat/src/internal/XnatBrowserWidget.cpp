#include "XnatBrowserWidget.h"

// XnatRestWidgets module includes
#include <XnatConnection.h>
#include <XnatDownloadManager.h>
#include <XnatException.h>
#include <XnatLoginDialog.h>
#include <XnatModel.h>
#include <XnatNameDialog.h>
#include <XnatNodeActivity.h>
#include <XnatNodeProperties.h>
#include <XnatSettings.h>
#include <XnatTreeView.h>
#include <XnatUploadManager.h>

// Qt includes
#include <QAction>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QMenu>
#include <QTextBrowser>

#include <mitkDataNodeFactory.h>

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
  QAction* saveAndUploadAction;
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

  if (!ui)
  {
    // Create UI
    ui = new Ui::XnatBrowserWidget();
    ui->setupUi(parent);

    d->uploadManager = new XnatUploadManager(ui->xnatTreeView);
    d->downloadManager = new XnatDownloadManager(ui->xnatTreeView);

    ui->middleButtonPanel->setCollapsed(true);

    ui->refreshButton->setEnabled(false);
    ui->downloadButton->setEnabled(false);
    ui->downloadAllButton->setEnabled(false);
    ui->importButton->setEnabled(false);
    ui->importAllButton->setEnabled(false);
    ui->uploadButton->setEnabled(false);
    ui->saveAndUploadButton->setEnabled(false);
    ui->createButton->setEnabled(false);
    ui->deleteButton->setEnabled(false);
    // Hide these buttons until thorougly tested:
    ui->uploadButton->setVisible(false);
    ui->saveAndUploadButton->setVisible(false);
    ui->createButton->setVisible(false);
    ui->deleteButton->setVisible(false);
	  
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

  if (ui)
  {
    delete d->downloadManager;
    delete d->uploadManager;
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
  d->downloadManager->setSettings(settings);
  d->uploadManager->setSettings(settings);
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
  connect(d->downloadAction, SIGNAL(triggered()), d->downloadManager, SLOT(downloadFile()));
  d->downloadAllAction = new QAction(tr("Download All"), this);
  connect(d->downloadAllAction, SIGNAL(triggered()), d->downloadManager, SLOT(downloadAllFiles()));
  d->importAction = new QAction(tr("Import"), this);
  connect(d->importAction, SIGNAL(triggered()), this, SLOT(importFile()));
  d->importAllAction = new QAction(tr("Import All"), this);
  connect(d->importAllAction, SIGNAL(triggered()), this, SLOT(importFiles()));
  d->uploadAction = new QAction(tr("Upload"), this);
  connect(d->uploadAction, SIGNAL(triggered()), d->uploadManager, SLOT(uploadFiles()));
  d->saveAndUploadAction = new QAction(tr("Save Data and Upload"), this);
  d->createAction = new QAction(tr("Create New"), this);
  connect(d->createAction, SIGNAL(triggered()), ui->xnatTreeView, SLOT(createNewRow()));
  d->deleteAction = new QAction(tr("Delete"), this);
  connect(d->deleteAction, SIGNAL(triggered()), ui->xnatTreeView, SLOT(deleteCurrentRow()));

  // create button widgets
  connect(ui->loginButton, SIGNAL(clicked()), this, SLOT(loginXnat()));
  connect(ui->refreshButton, SIGNAL(clicked()), ui->xnatTreeView, SLOT(refreshRows()));
  connect(ui->downloadButton, SIGNAL(clicked()), d->downloadManager, SLOT(downloadFile()));
  connect(ui->downloadAllButton, SIGNAL(clicked()), d->downloadManager, SLOT(downloadAllFiles()));
  connect(ui->importButton, SIGNAL(clicked()), this, SLOT(importFile()));
  connect(ui->importAllButton, SIGNAL(clicked()), this, SLOT(importFiles()));
  connect(ui->uploadButton, SIGNAL(clicked()), d->uploadManager, SLOT(uploadFiles()));
  connect(ui->saveAndUploadButton, SIGNAL(clicked()), d->saveAndUploadAction, SLOT(trigger()));
  connect(d->saveAndUploadAction, SIGNAL(changed()), this, SLOT(setSaveAndUploadButtonEnabled()));
  connect(ui->createButton, SIGNAL(clicked()), ui->xnatTreeView, SLOT(createNewRow()));
  connect(ui->deleteButton, SIGNAL(clicked()), ui->xnatTreeView, SLOT(deleteCurrentRow()));

  connect(ui->xnatTreeView, SIGNAL(clicked(const QModelIndex&)), this, SLOT(setButtonEnabled(const QModelIndex&)));

  ui->xnatTreeView->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(ui->xnatTreeView, SIGNAL(customContextMenuRequested(const QPoint&)),
          this, SLOT(showContextMenu(const QPoint&)));
}

void XnatBrowserWidget::loginXnat()
{
  Q_D(XnatBrowserWidget);

  // show dialog for user to login to XNAT
  XnatLoginDialog* loginDialog = new XnatLoginDialog(XnatConnectionFactory::instance(), this);
  loginDialog->setSettings(d->settings);
  if (loginDialog->exec())
  {
    // delete old connection
    if ( d->connection )
    {
      delete d->connection;
      d->connection = 0;
    }
    // get connection object
    d->connection = loginDialog->getConnection();

    ui->xnatTreeView->initialize(d->connection->getRoot());

    ui->downloadButton->setEnabled(false);
    ui->downloadAllButton->setEnabled(false);
    ui->importButton->setEnabled(false);
    ui->importAllButton->setEnabled(false);
    ui->uploadButton->setEnabled(false);
    ui->saveAndUploadButton->setEnabled(false);
    ui->createButton->setEnabled(false);
    ui->deleteButton->setEnabled(false);

    ui->refreshButton->setEnabled(true);
  }
  delete loginDialog;
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
  QString xnatFileNameTemp = QFileInfo(xnatFilename).fileName();
  QString tempWorkDirectory = d->settings->getWorkSubdirectory();
  d->downloadManager->silentlyDownloadFile(xnatFileNameTemp, tempWorkDirectory);

  // create list of files to open
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
    nodeFactory->SetFileName(filename.toStdString());
    nodeFactory->Update();
    mitk::DataNode::Pointer dataNode = nodeFactory->GetOutput();
    dataNode->SetName(xnatFilename.toStdString());
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
  QString xnatFileNameTemp = QFileInfo(xnatFilename).fileName();
  QString tempWorkDirectory = d->settings->getWorkSubdirectory();
  d->downloadManager->silentlyDownloadAllFiles(tempWorkDirectory);

  // create list of files to open
  QStringList files;
  collectImageFiles(tempWorkDirectory, files);

  if (d->dataStorage.IsNull())
  {
    return;
  }

  try
  {
    mitk::DataNodeFactory::Pointer nodeFactory = mitk::DataNodeFactory::New();
  //  mitk::FileReader::Pointer fileReader = mitk::FileReader::New();
    // determine reader type based on first file. For now, we are relying
    // on the user to avoid mixing file types.
    foreach (QString filename, files) {
      nodeFactory->SetFileName(filename.toStdString());
      nodeFactory->Update();
      mitk::DataNode::Pointer dataNode = nodeFactory->GetOutput();
      dataNode->SetName(xnatFilename.toStdString());
      d->dataStorage->Add(dataNode);
    }
  }
  catch (std::exception& exc)
  {
    MITK_INFO << "reading the image has failed";
  }
}

void XnatBrowserWidget::collectImageFiles(const QDir& tempWorkDirectory, QStringList& fileList)
{
  QFileInfoList files = tempWorkDirectory.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot, QDir::Name);
  bool first = true;
  foreach (QFileInfo file, files) {
    if (file.isDir()) {
      collectImageFiles(QDir(file.absoluteFilePath()), fileList);
    }
    else if (first)
    {
      fileList.push_back(file.filePath());
      first = false;
    }
  }
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
  ui->saveAndUploadButton->setEnabled((nodeProperties.receivesFiles() && d->saveAndUploadAction->isEnabled()));
  ui->createButton->setEnabled(nodeProperties.isModifiable());
  ui->deleteButton->setEnabled(nodeProperties.isDeletable());
}

void XnatBrowserWidget::setSaveAndUploadButtonEnabled()
{
  Q_D(XnatBrowserWidget);

  const XnatNodeProperties& nodeProperties = ui->xnatTreeView->currentNodeProperties();
  ui->saveAndUploadButton->setEnabled((nodeProperties.receivesFiles() && d->saveAndUploadAction->isEnabled()));
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

      if ( d->saveAndUploadAction->isEnabled() )
      {
        actions.append(d->saveAndUploadAction);
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
