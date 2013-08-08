#include "XnatDownloadManager.h"

#include <QDebug>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QTimer>
#include <QWidget>

extern "C"
{
#include "XnatRest.h"
}

#include <ctkXnatException.h>
#include <ctkXnatTreeModel.h>
#include <ctkXnatSettings.h>

#include "XnatDownloadDialog.h"
#include "XnatTreeView.h"

#include <mitkLogMacros.h>

class XnatDownloadManagerPrivate
{
public:
  XnatTreeView* xnatTreeView;
  XnatDownloadDialog* downloadDialog;

  ctkXnatSettings* settings;

  QString currDir;
  QString zipFilename;
  bool finished;
  unsigned long totalBytes;

  QString xnatFilename;
  QString outFilename;
  QString tempFilePath;
};

XnatDownloadManager::XnatDownloadManager(XnatTreeView* xnatTreeView)
: QObject(xnatTreeView)
, d_ptr(new XnatDownloadManagerPrivate())
{
  Q_D(XnatDownloadManager);

  d->xnatTreeView = xnatTreeView;
}

XnatDownloadManager::~XnatDownloadManager()
{
}

void XnatDownloadManager::setSettings(ctkXnatSettings* settings)
{
  Q_D(XnatDownloadManager);
  d->settings = settings;
}

void XnatDownloadManager::downloadFile()
{
  Q_D(XnatDownloadManager);

  // get name of file to be downloaded
  QModelIndex index = d->xnatTreeView->currentIndex();
  ctkXnatTreeModel* model = d->xnatTreeView->xnatModel();
  QString filename = model->data(index, Qt::DisplayRole).toString();
  if ( filename.isEmpty() )
  {
    return;
  }

  d->xnatFilename = QFileInfo(filename).fileName();

  d->currDir = d->settings->getDefaultDirectory();

  QString caption = tr("Save Downloaded File");
  QString dir = QFileInfo(d->currDir, d->xnatFilename).absoluteFilePath();
  // get output directory and filename from user
  QString userFilePath = QFileDialog::getSaveFileName(d->xnatTreeView, caption, dir);
  if ( userFilePath.isEmpty() )
  {
    return;
  }
  d->currDir = QFileInfo(userFilePath).absolutePath();
  d->outFilename = QFileInfo(userFilePath).fileName();

  // reset name of current directory to last directory viewed by user
  d->settings->setDefaultDirectory(d->currDir);

  // check if file exists with same name as file in XNAT
  d->tempFilePath = QString();
  if ( d->outFilename != d->xnatFilename )
  {
    QString xnatFilePath = QFileInfo(d->currDir, d->xnatFilename).absoluteFilePath();
    if ( QFile::exists(xnatFilePath) )
    {
      int i = 1;
      do
      {
        QString ver;
        ver.setNum(i++);
        d->tempFilePath = xnatFilePath;
        d->tempFilePath.append(".").append(ver);
      } while ( QFile::exists(d->tempFilePath) );

      if ( !QFile::rename(xnatFilePath, d->tempFilePath) )
      {
        QMessageBox::warning(d->xnatTreeView, tr("Downloaded File Error"), tr("Cannot rename existing file"));
        return;
      }
    }
  }

  // display download dialog
  d->downloadDialog = new XnatDownloadDialog(d->xnatTreeView);
  d->downloadDialog->show();

  QTimer::singleShot(0, this, SLOT(startDownload()));
}

void XnatDownloadManager::silentlyDownloadFile(const QString& fname, const QString& dir)
{
  Q_D(XnatDownloadManager);

  // initialize download variables
  d->xnatFilename = fname;
  d->outFilename = fname;
  d->currDir = dir;

  // check if file exists with same name as file in XNAT
  d->tempFilePath = QString();
  if ( d->outFilename != d->xnatFilename )
  {
    QString xnatFilePath = QFileInfo(d->currDir, d->xnatFilename).absoluteFilePath();
    if ( QFile::exists(xnatFilePath) )
    {
      int i = 1;
      do
      {
        QString ver;
        ver.setNum(i++);
        d->tempFilePath = xnatFilePath;
        d->tempFilePath.append(".").append(ver);
      } while ( QFile::exists(d->tempFilePath) );
      if ( !QFile::rename(xnatFilePath, d->tempFilePath) )
      {
        QMessageBox::warning(d->xnatTreeView, tr("Downloaded File Error"), tr("Cannot rename existing file"));
        return;
      }
    }
  }

  // display download dialog
  d->downloadDialog = new XnatDownloadDialog(d->xnatTreeView);
  d->downloadDialog->show();

  // start download of ZIP file
  d->zipFilename = QFileInfo(d->currDir, tr("xnat.file.zip")).absoluteFilePath();
  if ( !this->startFileDownload(d->zipFilename) )
  {
    d->downloadDialog->close();
    return;
  }

  // initialize download variables
  d->finished = false;
  d->totalBytes = 0;
  connect(this, SIGNAL(done()), this, SLOT(finishDownload()));

  downloadDataBlocking();
}

void XnatDownloadManager::startDownload()
{
  Q_D(XnatDownloadManager);

  // start download of ZIP file
  d->zipFilename = QFileInfo(d->currDir, tr("xnat.file.zip")).absoluteFilePath();
  if ( !this->startFileDownload(d->zipFilename) )
  {
    d->downloadDialog->close();
    return;
  }

  // initialize download variables
  d->finished = false;
  d->totalBytes = 0;
  connect(this, SIGNAL(done()), this, SLOT(finishDownload()));

  QTimer::singleShot(0, this, SLOT(downloadData()));
}

void XnatDownloadManager::startGroupDownload()
{
  Q_D(XnatDownloadManager);

  // start download of ZIP file
  d->zipFilename = QFileInfo(d->currDir, tr("xnat.group.zip")).absoluteFilePath();
  if ( !this->startFileDownload(d->zipFilename) )
  {
    d->downloadDialog->close();
    return;
  }

  // initialize download variables
  d->finished = false;
  d->totalBytes = 0;

  QTimer::singleShot(0, this, SLOT(downloadData()));
}

void XnatDownloadManager::downloadAllFiles()
{
  Q_D(XnatDownloadManager);

  // get name of file group to be downloaded
  QModelIndex index = d->xnatTreeView->selectionModel()->currentIndex();
  ctkXnatTreeModel* model = d->xnatTreeView->xnatModel();
//  QString groupname = model->name(index);
  QString groupname = model->data(index, Qt::DisplayRole).toString();
  if ( groupname.isEmpty() )
  {
    return;
  }

  // download files

  // initialize current directory
  d->currDir = d->settings->getDefaultDirectory();

  // get output directory from user
  QString outputDir = QFileDialog::getExistingDirectory(d->xnatTreeView, tr("Save Downloaded Files"), d->currDir);
  if ( outputDir.isEmpty() )
  {
    return;
  }
  d->currDir = outputDir;

  // reset name of current directory to last directory viewed by user
  d->settings->setDefaultDirectory(d->currDir);

  // display download dialog
  d->downloadDialog = new XnatDownloadDialog(d->xnatTreeView);
  d->downloadDialog->show();

  QTimer::singleShot(0, this, SLOT(startGroupDownload()));
}

void XnatDownloadManager::silentlyDownloadAllFiles(const QString& dir)
{
  Q_D(XnatDownloadManager);

  MITK_INFO << "XnatDownloadManager::silentlyDownloadAllFiles(const QString& dir) dir: " << dir.toStdString() << std::endl;
  // initialize download variables
//  d->xnatFilename = fname;
//  d->outFilename = fname;
  d->currDir = dir;

  // display download dialog
  d->downloadDialog = new XnatDownloadDialog(d->xnatTreeView);
  d->downloadDialog->show();

//  QTimer::singleShot(0, this, SLOT(startGroupDownload()));
  // start download of ZIP file
  d->zipFilename = QFileInfo(d->currDir, tr("xnat.file.zip")).absoluteFilePath();
  if (!this->startFileDownload(d->zipFilename))
  {
    MITK_INFO << "XnatDownloadManager::silentlyDownloadAllFiles(const QString& dir) download not started. zip file name: " << d->zipFilename.toStdString() << std::endl;
    d->downloadDialog->close();
    return;
  }

  // initialize download variables
  d->finished = false;
  d->totalBytes = 0;
  QObject::connect(this, SIGNAL(done()), this, SLOT(finishDownload()));

  this->downloadDataBlocking();
}

void XnatDownloadManager::downloadData()
{
  Q_D(XnatDownloadManager);

  unsigned long numBytes;
//  XnatRestStatus status;

  // check if user has canceled download
  if ( d->downloadDialog->wasDownloadCanceled() )
  {
    d->downloadDialog->close();
//    status = cancelXnatRestAsynTransfer();
//    if ( status != XNATREST_OK )
//    {
//      QMessageBox::warning(d->xnatTreeView, tr("Cancel File Download Error"), tr(getXnatRestStatusMsg(status)));
//    }
//    else if ( !QFile::remove(d->zipFilename) )
//    {
//      QMessageBox::warning(d->xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
//    }
    emit done();
    return;
  }

  // check if download is finished
  if ( d->finished == true )
  {
    d->downloadDialog->showUnzipInProgress();
    QTimer::singleShot(0, this, SLOT(unzipData()));
    return;
  }

  // download more data from XNAT
//  status = moveXnatRestAsynData(&numBytes, &d->finished);
  d->finished = true;
//  status = XNATREST_OK;
  numBytes = 0;

//  if ( status != XNATREST_OK )
//  {
//    d->downloadDialog->close();
//    QMessageBox::warning(d->xnatTreeView, tr("Download File Error"), tr(getXnatRestStatusMsg(status)));
//    emit done();
//    return;
//  }

  // update number of bytes downloaded
  if ( numBytes > 0 )
  {
    d->totalBytes += numBytes;
    d->downloadDialog->showBytesDownloaded(d->totalBytes);
  }

  QTimer::singleShot(0, this, SLOT(downloadData()));
}

void XnatDownloadManager::downloadDataBlocking()
{
  Q_D(XnatDownloadManager);
  MITK_INFO << "XnatDownloadManager::downloadDataBlocking()" << std::endl;

  unsigned long numBytes;

  while (true)
  {
    // check if user has canceled download
    if ( d->downloadDialog->wasDownloadCanceled() )
    {
      d->downloadDialog->close();
//      status = cancelXnatRestAsynTransfer();
//      if ( status != XNATREST_OK )
//      {
//        QMessageBox::warning(d->xnatTreeView, tr("Cancel File Download Error"), tr(getXnatRestStatusMsg(status)));
//      }
//      else if ( !QFile::remove(d->zipFilename) )
//      {
//        QMessageBox::warning(d->xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
//      }
      emit done();
      break;
    }

    // check if download is finished
    if ( d->finished == true )
    {
      d->downloadDialog->showUnzipInProgress();
      this->unzipData();
      break;
    }

    // download more data from XNAT
//    status = moveXnatRestAsynData(&numBytes, &d->finished);
    d->finished = true;
//    status = XNATREST_OK;
    numBytes = 0;

//    if ( status != XNATREST_OK )
//    {
//      d->downloadDialog->close();
//      QMessageBox::warning(d->xnatTreeView, tr("Download File Error"), tr(getXnatRestStatusMsg(status)));
//      emit done();
//      break;
//    }

    // update number of bytes downloaded
    if ( numBytes > 0 )
    {
      d->totalBytes += numBytes;
      d->downloadDialog->showBytesDownloaded(d->totalBytes);
    }
  }
}

void XnatDownloadManager::unzipData()
{
  MITK_INFO << "XnatDownloadManager::unzipData()" << std::endl;
  Q_D(XnatDownloadManager);

  // check if user has canceled download
  if ( d->downloadDialog->wasDownloadCanceled() )
  {
    d->downloadDialog->close();
    if ( !QFile::remove(d->zipFilename) )
    {
      QMessageBox::warning(d->xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
    }
    emit done();
    return;
  }

  MITK_INFO << "XnatDownloadManager::unzipData() zip file name: " << d->zipFilename.toStdString() << " ; current dir: " << d->currDir.toStdString() << std::endl;
  // unzip downloaded file
  XnatRestStatus status = unzipXnatRestFile(d->zipFilename.toAscii().constData(), d->currDir.toAscii().constData());

  // close dialog
  d->downloadDialog->close();

  if ( status != XNATREST_OK )    // check unzip status
  {
    QMessageBox::warning(d->xnatTreeView, tr("Unzip Downloaded File Error"), tr(getXnatRestStatusMsg(status)));
  }
  else if ( !QFile::remove(d->zipFilename) )    // delete zip file
  {
    QMessageBox::warning(d->xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
  }
  emit done();
}

void XnatDownloadManager::finishDownload()
{
  MITK_INFO << "XnatDownloadManager::finishDownload()" << std::endl;
  Q_D(XnatDownloadManager);

  QObject::disconnect(this, SIGNAL(done()), this, SLOT(finishDownload()));

  // change filename to name specified by user
  if ( d->outFilename != d->xnatFilename )
  {
    QString xnatFilePath = QFileInfo(d->currDir, d->xnatFilename).absoluteFilePath();
    if ( !QFile::rename(xnatFilePath, QFileInfo(d->currDir, d->outFilename).absoluteFilePath()) )
    {
      QMessageBox::warning(d->xnatTreeView, tr("Rename File Error"), tr("Cannot rename downloaded file"));
      return;
    }
    if ( !d->tempFilePath.isEmpty() )
    {
      if ( !QFile::rename(d->tempFilePath, xnatFilePath) )
      {
        QMessageBox::warning(d->xnatTreeView, tr("Rename File Error"), tr("Cannot rename existing file"));
      }
    }
  }
}

bool XnatDownloadManager::startFileDownload(const QString& zipFilename)
{
  Q_D(XnatDownloadManager);

  // start download of zip file
  try
  {
    QModelIndex index = d->xnatTreeView->selectionModel()->currentIndex();
    ctkXnatTreeModel* model = d->xnatTreeView->xnatModel();
    MITK_INFO << "XnatDownloadManager::startFileDownload(const QString& zipFilename) " << std::endl;
    model->downloadFile(index, zipFilename);
  }
  catch (ctkXnatException& e)
  {
    QMessageBox::warning(d->xnatTreeView, tr("Downloaded File Error"), tr(e.what()));
    return false;
  }

  return true;
}
