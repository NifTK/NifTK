/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatBrowserWidget.h"

#include <ctkXnatConnection.h>
#include <ctkXnatConnectionFactory.h>
#include <ctkXnatLoginDialog.h>
#include <ctkXnatObject.h>
#include <ctkXnatScanResource.h>
#include <ctkXnatSettings.h>

#include "XnatDownloadManager.h"
#include "ctkXnatTreeModel.h"
#include "XnatTreeView.h"

// Qt includes
#include <QAction>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QMenu>
#include <QTextBrowser>

#include <mitkDataNodeFactory.h>

#include <QDebug>

class XnatBrowserWidgetPrivate
{
public:
  ctkXnatSettings* settings;

  ctkXnatConnection* connection;
  XnatDownloadManager* downloadManager;

  QAction* downloadAction;
  QAction* downloadAllAction;
  QAction* importAction;
  QAction* importAllAction;

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

    d->downloadManager = new XnatDownloadManager(ui->xnatTreeView);

    ui->middleButtonPanel->setCollapsed(true);

    ui->refreshButton->setEnabled(false);
    ui->downloadButton->setEnabled(false);
    ui->downloadAllButton->setEnabled(false);
    ui->importButton->setEnabled(false);
    ui->importAllButton->setEnabled(false);

    // Create connections after setting defaults, so you don't trigger stuff when setting defaults.
    this->createConnections();
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
    delete ui;
  }
}

ctkXnatSettings* XnatBrowserWidget::settings() const
{
  Q_D(const XnatBrowserWidget);

  return d->settings;
}

void XnatBrowserWidget::setSettings(ctkXnatSettings* settings)
{
  Q_D(XnatBrowserWidget);
  d->settings = settings;
  d->downloadManager->setSettings(settings);
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

  // create button widgets
  QObject::connect(ui->loginButton, SIGNAL(clicked()), this, SLOT(loginXnat()));
  QObject::connect(ui->refreshButton, SIGNAL(clicked()), ui->xnatTreeView, SLOT(refreshRows()));
  QObject::connect(ui->downloadButton, SIGNAL(clicked()), d->downloadManager, SLOT(downloadFile()));
  QObject::connect(ui->downloadAllButton, SIGNAL(clicked()), d->downloadManager, SLOT(downloadAllFiles()));
  QObject::connect(ui->importButton, SIGNAL(clicked()), this, SLOT(importFile()));
  QObject::connect(ui->importAllButton, SIGNAL(clicked()), this, SLOT(importFiles()));

  QObject::connect(ui->xnatTreeView, SIGNAL(clicked(const QModelIndex&)), this, SLOT(setButtonEnabled(const QModelIndex&)));

  ui->xnatTreeView->setContextMenuPolicy(Qt::CustomContextMenu);
  QObject::connect(ui->xnatTreeView, SIGNAL(customContextMenuRequested(const QPoint&)),
          this, SLOT(showContextMenu(const QPoint&)));
}

void XnatBrowserWidget::loginXnat()
{
  Q_D(XnatBrowserWidget);

  // show dialog for user to login to XNAT
  ctkXnatLoginDialog* loginDialog = new ctkXnatLoginDialog(new ctkXnatConnectionFactory());
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

    ui->xnatTreeView->initialize(d->connection);

    ui->downloadButton->setEnabled(false);
    ui->downloadAllButton->setEnabled(false);
    ui->importButton->setEnabled(false);
    ui->importAllButton->setEnabled(false);

    ui->refreshButton->setEnabled(true);
  }
  delete loginDialog;
}

void XnatBrowserWidget::importFile()
{
  Q_D(XnatBrowserWidget);

  // get name of file to be downloaded
  QModelIndex index = ui->xnatTreeView->currentIndex();
  ctkXnatTreeModel* model = ui->xnatTreeView->xnatModel();
  QString xnatFilename = model->data(index, Qt::DisplayRole).toString();
  if (xnatFilename.isEmpty())
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
  ctkXnatTreeModel* model = ui->xnatTreeView->xnatModel();
  QString xnatFilename = model->data(index, Qt::DisplayRole).toString();
  if ( xnatFilename.isEmpty() )
  {
    return;
  }

  // download file
//  QString xnatFileNameTemp = QFileInfo(xnatFilename).fileName();
  QString tempWorkDirectory = d->settings->getWorkSubdirectory();
  d->downloadManager->silentlyDownloadAllFiles(tempWorkDirectory);

  // create list of files to open
  QStringList files;
  this->collectImageFiles(tempWorkDirectory, files);

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
    foreach (QString filename, files)
    {
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
  foreach (QFileInfo file, files)
  {
    if (file.isDir())
    {
      this->collectImageFiles(QDir(file.absoluteFilePath()), fileList);
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
  const ctkXnatObject::Pointer object = ui->xnatTreeView->getObject(index);

  ui->downloadButton->setEnabled(object->isFile());
  ui->downloadAllButton->setEnabled(this->holdsFiles(object));
  ui->importButton->setEnabled(object->isFile());
  ui->importAllButton->setEnabled(this->holdsFiles(object));
}

void XnatBrowserWidget::showContextMenu(const QPoint& position)
{
  Q_D(XnatBrowserWidget);

  const QModelIndex& index = ui->xnatTreeView->indexAt(position);
  if ( index.isValid() )
  {
    const ctkXnatObject::Pointer object = ui->xnatTreeView->getObject(index);
    QList<QAction*> actions;
    if ( object->isFile() )
    {
      actions.append(d->downloadAction);
    }
    if (this->holdsFiles(object))
    {
        actions.append(d->downloadAllAction);
        actions.append(d->importAllAction);
    }
    if ( object->isFile() )
    {
      actions.append(d->importAction);
    }
    if ( actions.count() > 0 )
    {
      QMenu::exec(actions, ui->xnatTreeView->mapToGlobal(position));
    }
  }
}

bool XnatBrowserWidget::holdsFiles(const ctkXnatObject::Pointer xnatObject) const
{
//  MITK_INFO << "XnatBrowserWidget::holdsFiles(const ctkXnatObject::Pointer xnatObject) const" << std::endl;
  ctkXnatObject* xnatObjectP = xnatObject.data();
  if (ctkXnatScanResource* scanResource = dynamic_cast<ctkXnatScanResource*>(xnatObjectP))
  {
//    scanResource->fetch();
//    MITK_INFO << "XnatBrowserWidget::holdsFiles(const ctkXnatObject::Pointer xnatObject) const This is a scan resource." << std::endl;
    if (scanResource->children().size() > 0)
    {
//      scanResource->reset();
//      MITK_INFO << "XnatBrowserWidget::holdsFiles(const ctkXnatObject::Pointer xnatObject) const It has children." << std::endl;
      return true;
    }
//    scanResource->reset();
  }
  return false;
}
