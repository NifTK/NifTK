#include "XnatUploadManager.h"

#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QTimer>

extern "C"
{
#include "XnatRest.h"
}

#include <ctkXnatException.h>
#include <ctkXnatSettings.h>

#include "XnatModel.h"
#include "XnatTreeView.h"
#include "XnatUploadDialog.h"


class XnatUploadManagerPrivate
{
public:
  XnatTreeView* xnatTreeView;
  XnatUploadDialog* uploadDialog;

  ctkXnatSettings* settings;

  QString currDir;
  QStringList userFilePaths;
  QString zipFilename;
  XnatRestAsynStatus finished;
  unsigned long totalBytes;
};

XnatUploadManager::XnatUploadManager(XnatTreeView* xnatTreeView)
: QObject(xnatTreeView)
, d_ptr(new XnatUploadManagerPrivate())
{
  Q_D(XnatUploadManager);

  d->xnatTreeView = xnatTreeView;
}

XnatUploadManager::~XnatUploadManager()
{
}

void XnatUploadManager::setSettings(ctkXnatSettings* settings)
{
  Q_D(XnatUploadManager);
  d->settings = settings;
}

void XnatUploadManager::uploadFiles()
{
  Q_D(XnatUploadManager);

  // get names of files to be uploaded
  if ( !getFilenames() )
  {
    return;
  }

  // display upload dialog
  d->uploadDialog = new XnatUploadDialog(d->xnatTreeView);
  d->uploadDialog->show();

  // zip files for upload
  QTimer::singleShot(0, this, SLOT(zipFiles()));
}

void XnatUploadManager::uploadSavedData(const QString& dir)
{
  Q_D(XnatUploadManager);

  // set current directory
  d->currDir = dir;

  // get names of files to be uploaded
  d->userFilePaths = QDir(dir).entryList(QDir::Files);

  // display upload dialog
  d->uploadDialog = new XnatUploadDialog(d->xnatTreeView);
  d->uploadDialog->show();

  // zip files for upload
  QTimer::singleShot(0, this, SLOT(zipFiles()));
}

bool XnatUploadManager::getFilenames()
{
  Q_D(XnatUploadManager);

  // get current directory
  d->currDir = d->settings->getDefaultDirectory();

  // display file dialog to get names of files to be uploaded
  QFileDialog fileNameDialog(d->xnatTreeView, tr("Select File to Upload"), d->currDir);
  fileNameDialog.setFileMode(QFileDialog::ExistingFiles);
  // handle multiple filenames in multiple subdirectories

  // get names of files to be uploaded
  if ( fileNameDialog.exec() )
  {
    d->userFilePaths = fileNameDialog.selectedFiles();
  }
  else
  {
    return false;
  }

  if ( d->userFilePaths.size() == 0 )
  {
    QMessageBox::warning(d->xnatTreeView, tr("File to Upload"), tr("No file selected"));
    return false;
  }

  // reset name of current directory to last directory viewed by user
  d->currDir = fileNameDialog.directory().absolutePath();
  d->settings->setDefaultDirectory(d->currDir);

  return true;
}

void XnatUploadManager::zipFiles()
{
  Q_D(XnatUploadManager);

  // get names of files to be uploaded
  int numFilenames = d->userFilePaths.size();
  char **filenames = new char*[numFilenames];
  for ( int i = 0 ; i < numFilenames ; i++ )
  {
    QByteArray tmp = QFileInfo(d->userFilePaths.at(i)).fileName().toAscii();
    filenames[i] = new char[tmp.size() + 1];
    strcpy(filenames[i], tmp.data());
  }

  // zip files to be uploaded
  d->zipFilename = QFileInfo(d->currDir, tr("xnat.upload.zip")).absoluteFilePath();
  XnatRestStatus status = zipXnatRestFile(d->zipFilename.toAscii().constData(),
                                          d->currDir.toAscii().constData(), numFilenames, filenames);
  for ( int i = 0 ; i < numFilenames ; i++ )
  {
    delete [] filenames[i];
  }
  delete [] filenames;
  if ( status != XNATREST_OK )
  {
    d->uploadDialog->close();
    QMessageBox::warning(d->xnatTreeView, tr("Zip Upload File Error"), tr(getXnatRestStatusMsg(status)));
    return;
  }

  // check if user has canceled upload
  if ( d->uploadDialog->wasUploadCanceled() )
  {
    d->uploadDialog->close();
    if ( !QFile::remove(d->zipFilename) )
    {
      QMessageBox::warning(d->xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
    }
    return;
  }

  d->uploadDialog->showUploadStarting();
  QTimer::singleShot(0, this, SLOT(startUpload()));
}

void XnatUploadManager::startUpload()
{
  Q_D(XnatUploadManager);

  // start upload of ZIP file
  if ( !this->startFileUpload(d->zipFilename) )
  {
    d->uploadDialog->close();
    return;
  }

  // initialize variables for uploading data
  d->finished = XNATRESTASYN_NOTDONE;
  d->totalBytes = 0;

  QTimer::singleShot(0, this, SLOT(uploadData()));
}

void XnatUploadManager::uploadData()
{
  Q_D(XnatUploadManager);

  unsigned long numBytes;
  XnatRestStatus status;

  // check if user has canceled upload
  if ( d->uploadDialog->wasUploadCanceled() )
  {
    d->uploadDialog->close();
//    status = cancelXnatRestAsynTransfer();
//    if ( status != XNATREST_OK )
//    {
//      QMessageBox::warning(d->xnatTreeView, tr("Cancel File Upload Error"), tr(getXnatRestStatusMsg(status)));
//    }
//    else if ( !QFile::remove(d->zipFilename) )
//    {
//      QMessageBox::warning(d->xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
//    }
    return;
  }

  // upload more data to XNAT
//  status = moveXnatRestAsynData(&numBytes, &d->finished);
  d->finished = XNATRESTASYN_DONE;
  status = XNATREST_OK;
  numBytes = 0;

  if ( status != XNATREST_OK )
  {
    d->uploadDialog->close();
    QMessageBox::warning(d->xnatTreeView, tr("Upload File Error"), tr(getXnatRestStatusMsg(status)));
    return;
  }

  // check if upload is finished
  if ( d->finished == XNATRESTASYN_DONE )
  {
    d->uploadDialog->close();
    if ( !QFile::remove(d->zipFilename) )
    {
      QMessageBox::warning(d->xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
    }
    d->xnatTreeView->refreshRows();
    return;
  }

  // update number of bytes uploaded
  if ( numBytes > 0 )
  {
    d->totalBytes += numBytes;
    d->uploadDialog->showBytesUploaded(d->totalBytes);
  }

  QTimer::singleShot(0, this, SLOT(uploadData()));
}

bool XnatUploadManager::startFileUpload(const QString& zipFilename)
{
  Q_D(XnatUploadManager);

  // start upload of zip file
  try
  {
    QModelIndex index = d->xnatTreeView->selectionModel()->currentIndex();
    XnatModel* model = d->xnatTreeView->xnatModel();
    model->uploadFile(index, zipFilename);
  }
  catch (ctkXnatException& e)
  {
    QMessageBox::warning(d->xnatTreeView, tr("Upload Files Error"), tr(e.what()));
    return false;
  }

  return true;
}

void XnatUploadManager::refreshRows()
{
  Q_D(XnatUploadManager);

  d->xnatTreeView->refreshRows();
}
