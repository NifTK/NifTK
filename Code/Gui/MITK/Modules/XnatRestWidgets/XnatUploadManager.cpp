/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatUploadManager.h"

#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QTimer>

#include "XnatException.h"
#include "XnatModel.h"
#include "XnatSettings.h"
#include "XnatTreeView.h"
#include "XnatUploadDialog.h"


XnatUploadManager::XnatUploadManager(XnatTreeView* xnatTreeView)
: QObject(xnatTreeView)
, xnatTreeView(xnatTreeView)
{
}

void XnatUploadManager::setSettings(XnatSettings* settings)
{
  this->settings = settings;
}

void XnatUploadManager::uploadFiles()
{
  // get names of files to be uploaded
  if ( !getFilenames() )
  {
    return;
  }

  // display upload dialog
  uploadDialog = new XnatUploadDialog(xnatTreeView);
  uploadDialog->show();

  // zip files for upload
  QTimer::singleShot(0, this, SLOT(zipFiles()));
}

void XnatUploadManager::uploadSavedData(const QString& dir)
{
  // set current directory
  currDir = dir;

  // get names of files to be uploaded
  userFilePaths = QDir(dir).entryList(QDir::Files);

  // display upload dialog
  uploadDialog = new XnatUploadDialog(xnatTreeView);
  uploadDialog->show();

  // zip files for upload
  QTimer::singleShot(0, this, SLOT(zipFiles()));
}

bool XnatUploadManager::getFilenames()
{
  // get current directory
  currDir = settings->getDefaultDirectory();

  // display file dialog to get names of files to be uploaded
  QFileDialog fileNameDialog(xnatTreeView, tr("Select File to Upload"), currDir);
  fileNameDialog.setFileMode(QFileDialog::ExistingFiles);
  // handle multiple filenames in multiple subdirectories

  // get names of files to be uploaded
  if ( fileNameDialog.exec() )
  {
    userFilePaths = fileNameDialog.selectedFiles();
  }
  else
  {
    return false;
  }

  if ( userFilePaths.size() == 0 )
  {
    QMessageBox::warning(xnatTreeView, tr("File to Upload"), tr("No file selected"));
    return false;
  }

  // reset name of current directory to last directory viewed by user
  currDir = fileNameDialog.directory().absolutePath();
  settings->setDefaultDirectory(currDir);

  return true;
}

void XnatUploadManager::zipFiles()
{
  // get names of files to be uploaded
  int numFilenames = userFilePaths.size();
  char **filenames = new char*[numFilenames];
  for ( int i = 0 ; i < numFilenames ; i++ )
  {
    QByteArray tmp = QFileInfo(userFilePaths.at(i)).fileName().toAscii();
    filenames[i] = new char[tmp.size() + 1];
    strcpy(filenames[i], tmp.data());
  }

  // zip files to be uploaded
  zipFilename = QFileInfo(currDir, tr("xnat.upload.zip")).absoluteFilePath();
  XnatRestStatus status = zipXnatRestFile(zipFilename.toAscii().constData(),
                                          currDir.toAscii().constData(), numFilenames, filenames);
  for ( int i = 0 ; i < numFilenames ; i++ )
  {
    delete [] filenames[i];
  }
  delete [] filenames;
  if ( status != XNATREST_OK )
  {
    uploadDialog->close();
    QMessageBox::warning(xnatTreeView, tr("Zip Upload File Error"), tr(getXnatRestStatusMsg(status)));
    return;
  }

  // check if user has canceled upload
  if ( uploadDialog->wasUploadCanceled() )
  {
    uploadDialog->close();
    if ( !QFile::remove(zipFilename) )
    {
      QMessageBox::warning(xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
    }
    return;
  }

  uploadDialog->showUploadStarting();
  QTimer::singleShot(0, this, SLOT(startUpload()));
}

void XnatUploadManager::startUpload()
{
  // start upload of ZIP file
  if ( !this->startFileUpload(zipFilename) )
  {
    uploadDialog->close();
    return;
  }

  // initialize variables for uploading data
  finished = XNATRESTASYN_NOTDONE;
  totalBytes = 0;

  QTimer::singleShot(0, this, SLOT(uploadData()));
}

void XnatUploadManager::uploadData()
{
  unsigned long numBytes;
  XnatRestStatus status;

  // check if user has canceled upload
  if ( uploadDialog->wasUploadCanceled() )
  {
    uploadDialog->close();
    status = cancelXnatRestAsynTransfer();
    if ( status != XNATREST_OK )
    {
      QMessageBox::warning(xnatTreeView, tr("Cancel File Upload Error"), tr(getXnatRestStatusMsg(status)));
    }
    else if ( !QFile::remove(zipFilename) )
    {
      QMessageBox::warning(xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
    }
    return;
  }

  // upload more data to XNAT
  status = moveXnatRestAsynData(&numBytes, &finished);
  if ( status != XNATREST_OK )
  {
    uploadDialog->close();
    QMessageBox::warning(xnatTreeView, tr("Upload File Error"), tr(getXnatRestStatusMsg(status)));
    return;
  }

  // check if upload is finished
  if ( finished == XNATRESTASYN_DONE )
  {
    uploadDialog->close();
    if ( !QFile::remove(zipFilename) )
    {
      QMessageBox::warning(xnatTreeView, tr("Delete Zip File Error"), tr("Cannot delete zip file"));
    }
    xnatTreeView->refreshRows();
    return;
  }

  // update number of bytes uploaded
  if ( numBytes > 0 )
  {
    totalBytes += numBytes;
    uploadDialog->showBytesUploaded(totalBytes);
  }

  QTimer::singleShot(0, this, SLOT(uploadData()));
}

bool XnatUploadManager::startFileUpload(const QString& zipFilename)
{
  // start upload of zip file
  try
  {
    QModelIndex index = xnatTreeView->selectionModel()->currentIndex();
    XnatModel* model = xnatTreeView->xnatModel();
    model->uploadFile(index, zipFilename);
  }
  catch (XnatException& e)
  {
    QMessageBox::warning(xnatTreeView, tr("Upload Files Error"), tr(e.what()));
    return false;
  }

  return true;
}

void XnatUploadManager::refreshRows()
{
  xnatTreeView->refreshRows();
}
