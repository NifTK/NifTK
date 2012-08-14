#include "XnatDownloadManager.h"

#include <QFile>
#include <QFileInfo>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>

#include "XnatBrowserWidget.h"
#include "XnatDownloadDialog.h"
#include "XnatPluginSettings.h"

XnatDownloadManager::XnatDownloadManager(XnatBrowserWidget* b)
: QObject(), browser(b)
{
  settings = browser->settings();
}

void XnatDownloadManager::downloadFile(const QString& fname)
{
  // initialize download variables
  xnatFilename = fname;
  currDir = settings->getDefaultDirectory();

  QWidget* parent = browser;
  QString caption = tr("Save Downloaded File");
  QString dir = QFileInfo(currDir, xnatFilename).absoluteFilePath();
  // get output directory and filename from user
  QString userFilePath = QFileDialog::getSaveFileName(parent, caption, dir);
  if ( userFilePath.isEmpty() )
  {
    return;
  }
  currDir = QFileInfo(userFilePath).absolutePath();
  outFilename = QFileInfo(userFilePath).fileName();

  // reset name of current directory to last directory viewed by user
  settings->setDefaultDirectory(currDir);

  // check if file exists with same name as file in XNAT
  tempFilePath = QString();
  if ( outFilename != xnatFilename )
  {
    QString xnatFilePath = QFileInfo(currDir, xnatFilename).absoluteFilePath();
    if ( QFile::exists(xnatFilePath) )
    {
      int i = 1;
      do
      {
        QString ver;
        ver.setNum(i++);
        tempFilePath = xnatFilePath;
        tempFilePath.append(".").append(ver);
      } while ( QFile::exists(tempFilePath) );

      if ( !QFile::rename(xnatFilePath, tempFilePath) )
      {
        QMessageBox::warning(parent, tr("Downloaded File Error"), tr("Cannot rename existing file"));
        return;
      }
    }
  }

  // display download dialog
  downloadDialog = new XnatDownloadDialog(parent);
  downloadDialog->show();

  QTimer::singleShot(0, this, SLOT(startDownload()));
}

void XnatDownloadManager::silentlyDownloadFile(const QString& fname, const QString& dir)
{
  QWidget* parent = browser;

  // initialize download variables
  xnatFilename = fname;
  outFilename = fname;
  currDir = dir;

  // check if file exists with same name as file in XNAT
  tempFilePath = QString();
  if ( outFilename != xnatFilename )
  {
    QString xnatFilePath = QFileInfo(currDir, xnatFilename).absoluteFilePath();
    if ( QFile::exists(xnatFilePath) )
    {
      int i = 1;
      do
      {
        QString ver;
        ver.setNum(i++);
        tempFilePath = xnatFilePath;
        tempFilePath.append(".").append(ver);
      } while ( QFile::exists(tempFilePath) );
      if ( !QFile::rename(xnatFilePath, tempFilePath) )
      {
        QMessageBox::warning(parent, tr("Downloaded File Error"), tr("Cannot rename existing file"));
        return;
      }
    }
  }

  // display download dialog
  downloadDialog = new XnatDownloadDialog(parent);
  downloadDialog->show();

  // start download of ZIP file
  zipFilename = QFileInfo(currDir, tr("xnat.file.zip")).absoluteFilePath();
  if ( !browser->startFileDownload(zipFilename) )
  {
    downloadDialog->close();
    return;
  }

  // initialize download variables
  finished = XNATRESTASYN_NOTDONE;
  totalBytes = 0;
  connect(this, SIGNAL(done()), this, SLOT(finishDownload()));

  downloadDataBlocking();
}

void XnatDownloadManager::startDownload()
{
  // start download of ZIP file
  zipFilename = QFileInfo(currDir, tr("xnat.file.zip")).absoluteFilePath();
  if ( !browser->startFileDownload(zipFilename) )
  {
    downloadDialog->close();
    return;
  }

  // initialize download variables
  finished = XNATRESTASYN_NOTDONE;
  totalBytes = 0;
  connect(this, SIGNAL(done()), this, SLOT(finishDownload()));

  QTimer::singleShot(0, this, SLOT(downloadData()));
}

void XnatDownloadManager::downloadAllFiles()
{
  // initialize current directory
  currDir = settings->getDefaultDirectory();

  // get output directory from user
  QString outputDir = QFileDialog::getExistingDirectory(browser, tr("Save Downloaded Files"), currDir);
  if ( outputDir.isEmpty() )
  {
    return;
  }
  currDir = outputDir;

  // reset name of current directory to last directory viewed by user
  settings->setDefaultDirectory(currDir);

  // display download dialog
  downloadDialog = new XnatDownloadDialog(browser);
  downloadDialog->show();

  QTimer::singleShot(0, this, SLOT(startGroupDownload()));
}

void XnatDownloadManager::startGroupDownload()
{
  // start download of ZIP file
  zipFilename = QFileInfo(currDir, tr("xnat.group.zip")).absoluteFilePath();
  if ( !browser->startFileGroupDownload(zipFilename) )
  {
    downloadDialog->close();
    return;
  }

  // initialize download variables
  finished = XNATRESTASYN_NOTDONE;
  totalBytes = 0;

  QTimer::singleShot(0, this, SLOT(downloadData()));
}

void XnatDownloadManager::downloadData()
{
  unsigned long numBytes;
  XnatRestStatus status;

  // check if user has canceled download
  if ( downloadDialog->wasDownloadCanceled() )
  {
    downloadDialog->close();
    status = cancelXnatRestAsynTransfer();
    if ( status != XNATREST_OK )
    {
      QMessageBox::warning(browser, tr("Cancel File Download Error"), tr(getXnatRestStatusMsg(status)));
    }
    else if ( !QFile::remove(zipFilename) )
    {
      QMessageBox::warning(browser, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
    }
    emit done();
    return;
  }

  // check if download is finished
  if ( finished == XNATRESTASYN_DONE )
  {
    downloadDialog->showUnzipInProgress();
    QTimer::singleShot(0, this, SLOT(unzipData()));
    return;
  }

  // download more data from XNAT
  status = moveXnatRestAsynData(&numBytes, &finished);
  if ( status != XNATREST_OK )
  {
    downloadDialog->close();
    QMessageBox::warning(browser, tr("Download File Error"), tr(getXnatRestStatusMsg(status)));
    emit done();
    return;
  }

  // update number of bytes downloaded
  if ( numBytes > 0 )
  {
    totalBytes += numBytes;
    downloadDialog->showBytesDownloaded(totalBytes);
  }

  QTimer::singleShot(0, this, SLOT(downloadData()));
}

void XnatDownloadManager::downloadDataBlocking()
{
  unsigned long numBytes;
  XnatRestStatus status;

  // check if user has canceled download
  if ( downloadDialog->wasDownloadCanceled() )
  {
    downloadDialog->close();
    status = cancelXnatRestAsynTransfer();
    if ( status != XNATREST_OK )
    {
      QMessageBox::warning(browser, tr("Cancel File Download Error"), tr(getXnatRestStatusMsg(status)));
    }
    else if ( !QFile::remove(zipFilename) )
    {
      QMessageBox::warning(browser, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
    }
    emit done();
    return;
  }

  // check if download is finished
  if ( finished == XNATRESTASYN_DONE )
  {
    downloadDialog->showUnzipInProgress();
    unzipData();
    return;
  }

  // download more data from XNAT
  status = moveXnatRestAsynData(&numBytes, &finished);
  if ( status != XNATREST_OK )
  {
    downloadDialog->close();
    QMessageBox::warning(browser, tr("Download File Error"), tr(getXnatRestStatusMsg(status)));
    emit done();
    return;
  }

  // update number of bytes downloaded
  if ( numBytes > 0 )
  {
    totalBytes += numBytes;
    downloadDialog->showBytesDownloaded(totalBytes);
  }

  downloadDataBlocking();
}

void XnatDownloadManager::unzipData()
{
  // check if user has canceled download
  if ( downloadDialog->wasDownloadCanceled() )
  {
    downloadDialog->close();
    if ( !QFile::remove(zipFilename) )
    {
      QMessageBox::warning(browser, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
    }
    emit done();
    return;
  }

  // unzip downloaded file
  XnatRestStatus status = unzipXnatRestFile(zipFilename.toAscii().constData(), currDir.toAscii().constData());

  // close dialog
  downloadDialog->close();

  if ( status != XNATREST_OK )    // check unzip status
  {
    QMessageBox::warning(browser, tr("Unzip Downloaded File Error"), tr(getXnatRestStatusMsg(status)));
  }
  else if ( !QFile::remove(zipFilename) )    // delete zip file
  {
    QMessageBox::warning(browser, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
  }
  emit done();
}

void XnatDownloadManager::finishDownload()
{
  disconnect(this, SIGNAL(done()), this, SLOT(finishDownload()));

  // change filename to name specified by user
  if ( outFilename != xnatFilename )
  {
    QString xnatFilePath = QFileInfo(currDir, xnatFilename).absoluteFilePath();
    if ( !QFile::rename(xnatFilePath, QFileInfo(currDir, outFilename).absoluteFilePath()) )
    {
      QMessageBox::warning(browser, tr("Rename File Error"), tr("Cannot rename downloaded file"));
      return;
    }
    if ( !tempFilePath.isEmpty() )
    {
      if ( !QFile::rename(tempFilePath, xnatFilePath) )
      {
        QMessageBox::warning(browser, tr("Rename File Error"), tr("Cannot rename existing file"));
      }
    }
  }
}

