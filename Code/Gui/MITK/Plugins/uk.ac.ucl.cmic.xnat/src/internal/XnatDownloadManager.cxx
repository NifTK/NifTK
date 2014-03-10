#include "XnatDownloadManager.h"

#include <QDebug>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QTimer>
#include <QWidget>
#include <QDataStream>

#include <ctkXnatException.h>
#include <ctkXnatTreeModel.h>
#include <ctkXnatSettings.h>

#include "XnatDownloadDialog.h"
#include "XnatTreeView.h"

/// Note: This is needed for miniz.
#if defined(__GNUC__)
  // Ensure we get the 64-bit variants of the CRT's file I/O calls
  #ifndef _FILE_OFFSET_BITS
    #define _FILE_OFFSET_BITS 64
  #endif
  #ifndef _LARGEFILE64_SOURCE
    #define _LARGEFILE64_SOURCE 1
  #endif
#endif

extern "C"
{
#include "miniz.c"
}

class XnatDownloadManagerPrivate
{
public:
  XnatTreeView* xnatTreeView;
  XnatDownloadDialog* downloadDialog;

  ctkXnatSettings* settings;

  QString currDir;
  QString zipFileName;
  bool finished;
  unsigned long totalBytes;

  QString xnatFileName;
  QString outFileName;
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
  QString fileName = model->data(index, Qt::DisplayRole).toString();
  if (fileName.isEmpty())
  {
    return;
  }

  d->xnatFileName = QFileInfo(fileName).fileName();

  d->currDir = d->settings->defaultDirectory();

  QString caption = tr("Save Downloaded File");
  QString dir = QFileInfo(d->currDir, d->xnatFileName).absoluteFilePath();
  // get output directory and filename from user
  QString userFilePath = QFileDialog::getSaveFileName(d->xnatTreeView, caption, dir);
  if (userFilePath.isEmpty())
  {
    return;
  }
  d->currDir = QFileInfo(userFilePath).absolutePath();
  d->outFileName = QFileInfo(userFilePath).fileName();

  // reset name of current directory to last directory viewed by user
  d->settings->setDefaultDirectory(d->currDir);

  // check if file exists with same name as file in XNAT
  d->tempFilePath = QString();
  if (d->outFileName != d->xnatFileName)
  {
    QString xnatFilePath = QFileInfo(d->currDir, d->xnatFileName).absoluteFilePath();
    if ( QFile::exists(xnatFilePath) )
    {
      int i = 1;
      do
      {
        QString ver;
        ver.setNum(i++);
        d->tempFilePath = xnatFilePath;
        d->tempFilePath.append(".").append(ver);
      } while (QFile::exists(d->tempFilePath));

      if (!QFile::rename(xnatFilePath, d->tempFilePath))
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

void XnatDownloadManager::silentlyDownloadFile(const QString& fileName, const QString& dir)
{
  Q_D(XnatDownloadManager);

  // initialize download variables
  d->xnatFileName = fileName;
  d->outFileName = fileName;
  d->currDir = dir;

  // check if file exists with same name as file in XNAT
  d->tempFilePath = QString();
  if ( d->outFileName != d->xnatFileName )
  {
    QString xnatFilePath = QFileInfo(d->currDir, d->xnatFileName).absoluteFilePath();
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
  d->zipFileName = QFileInfo(d->currDir, d->outFileName).absoluteFilePath();
  if (!this->startFileDownload(d->zipFileName))
  {
    d->downloadDialog->close();
    return;
  }

  // initialize download variables
  d->finished = false;
  d->totalBytes = 0;
  this->connect(this, SIGNAL(done()), SLOT(finishDownload()));

  this->downloadDataBlocking(false);
}

void XnatDownloadManager::startDownload()
{
  Q_D(XnatDownloadManager);

  // start download of ZIP file
  d->zipFileName = QFileInfo(d->currDir, d->outFileName).absoluteFilePath();
  if (!this->startFileDownload(d->zipFileName))
  {
    d->downloadDialog->close();
    return;
  }

  // initialize download variables
  d->finished = false;
  d->totalBytes = 0;
  this->connect(this, SIGNAL(done()), SLOT(finishDownload()));

  QTimer::singleShot(0, this, SLOT(downloadData()));
}

void XnatDownloadManager::startGroupDownload()
{
  Q_D(XnatDownloadManager);

  // start download of ZIP file
  d->zipFileName = QFileInfo(d->currDir, tr("xnat.group.zip")).absoluteFilePath();
  if ( !this->startFileDownload(d->zipFileName) )
  {
    d->downloadDialog->close();
    return;
  }

  // initialize download variables
  d->finished = false;
  d->totalBytes = 0;

  QTimer::singleShot(0, this, SLOT(downloadDataAndUnzip()));
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
  d->currDir = d->settings->defaultDirectory();

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

  // initialize download variables
//  d->xnatFilename = fname;
//  d->outFilename = fname;
  d->currDir = dir;

  // display download dialog
  d->downloadDialog = new XnatDownloadDialog(d->xnatTreeView);
  d->downloadDialog->show();

//  QTimer::singleShot(0, this, SLOT(startGroupDownload()));
  // start download of ZIP file
  d->zipFileName = QFileInfo(d->currDir, tr("xnat.file.zip")).absoluteFilePath();
  if (!this->startFileDownload(d->zipFileName))
  {
    d->downloadDialog->close();
    return;
  }

  // initialize download variables
  d->finished = false;
  d->totalBytes = 0;
  this->connect(this, SIGNAL(done()), SLOT(finishDownload()));

  this->downloadDataBlocking(true);
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
    d->downloadDialog->close();
    emit done();
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

void XnatDownloadManager::downloadDataAndUnzip()
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

  QTimer::singleShot(0, this, SLOT(downloadDataAndUnzip()));
}

void XnatDownloadManager::downloadDataBlocking(bool unzip)
{
  Q_D(XnatDownloadManager);

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
      if (unzip)
      {
        d->downloadDialog->showUnzipInProgress();
        this->unzipData();
      }
      else
      {
        // close dialog
        d->downloadDialog->close();
        emit done();
      }
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
  Q_D(XnatDownloadManager);

  // check if user has canceled download
  if ( d->downloadDialog->wasDownloadCanceled() )
  {
    d->downloadDialog->close();
    if ( !QFile::remove(d->zipFileName) )
    {
      QMessageBox::warning(d->xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
    }
    emit done();
    return;
  }

  // unzip downloaded file
  QStringList files = this->ExtractFile(d->zipFileName, d->currDir);

  // close dialog
  d->downloadDialog->close();

  if (files.isEmpty())    // check unzip status
  {
    QMessageBox::warning(d->xnatTreeView, tr("Unzip Downloaded File Error"), tr("Failed to extract the archive."));
  }
  else if (!QFile::remove(d->zipFileName))    // delete zip file
  {
    QMessageBox::warning(d->xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
  }
  emit done();
}


QStringList XnatDownloadManager::ExtractFile(QString zipFileName, QString directory)
{
  QStringList result;

  QDir dir(directory);

  mz_zip_archive zipArchive;
  std::memset(&zipArchive, 0, sizeof(zipArchive));

  mz_bool status = mz_zip_reader_init_file(&zipArchive, zipFileName.toAscii().data(), 0);
  if (!status)
  {
    throw "mz_zip_reader_init_file() failed!\n";
  }

  mz_uint fileNumber = mz_zip_reader_get_num_files(&zipArchive);

  for (mz_uint i = 0; i < fileNumber; ++i)
  {
    std::size_t uncompressedSize;
    void* p = mz_zip_reader_extract_to_heap(&zipArchive, i, &uncompressedSize, 0);

    mz_zip_archive_file_stat fileStat;
    mz_zip_reader_file_stat(&zipArchive, i, &fileStat);

    if (mz_zip_reader_is_file_a_directory(&zipArchive, i))
    {
      dir.mkpath(fileStat.m_filename);
    }
    else
    {
      if (!p)
      {
        mz_zip_reader_end(&zipArchive);
        throw "mz_zip_reader_extract_file_to_heap() failed!\n";
      }

      QFileInfo extractedFile(dir, fileStat.m_filename);
      QDir extractedFileDir = extractedFile.dir();
      if (!extractedFileDir.exists())
      {
        dir.mkpath(extractedFileDir.path());
      }
      QFile file(extractedFile.absoluteFilePath());
      file.open(QIODevice::WriteOnly);
      QDataStream stream(&file);
      /// Magic number alert!
      /// 4 should be replaced with some of the offset literals, maybe MZ_ZIP_ECDH_NUM_THIS_DISK_OFS?
      stream.writeBytes((const char*)p + 4, uncompressedSize - 4);
      file.close();
      result << extractedFile.absoluteFilePath();
    }

    mz_free(p);
  }

  mz_zip_reader_end(&zipArchive);

  return result;
}

void XnatDownloadManager::finishDownload()
{
  Q_D(XnatDownloadManager);

  QObject::disconnect(this, SIGNAL(done()), this, SLOT(finishDownload()));

  // change filename to name specified by user
  if ( d->outFileName != d->xnatFileName )
  {
    QString xnatFilePath = QFileInfo(d->currDir, d->xnatFileName).absoluteFilePath();
    if ( !QFile::rename(xnatFilePath, QFileInfo(d->currDir, d->outFileName).absoluteFilePath()) )
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

bool XnatDownloadManager::startFileDownload(const QString& fileName)
{
  Q_D(XnatDownloadManager);

  // start download of file
  try
  {
    QModelIndex index = d->xnatTreeView->selectionModel()->currentIndex();
    ctkXnatTreeModel* model = d->xnatTreeView->xnatModel();
    model->downloadFile(index, fileName);
  }
  catch (ctkXnatException& e)
  {
    QMessageBox::warning(d->xnatTreeView, tr("Downloaded File Error"), tr(e.what()));
    return false;
  }

  return true;
}
