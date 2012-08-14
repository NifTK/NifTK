#include "pqActiveObjects.h"
#include "pqApplicationCore.h"
#include "pqCoreUtilities.h"
#include "pqFileDialog.h"
#include "pqOptions.h"
#include "pqOutputPort.h"
#include "pqPipelineSource.h"
#include "pqServer.h"
#include "pqTestUtility.h"
#include "pqWriterDialog.h"
#include "vtkSmartPointer.h"
#include "vtkSMProxyManager.h"
#include "vtkSMSourceProxy.h"
#include "vtkSMWriterFactory.h"

#include <QMessageBox>
#include <QDebug>
#include <QFileInfo>
#include <QDir>
#include <QUuid>

#include "XnatBrowserSettings.h"
#include "XnatFileNameTypeDialog.h"
#include "XnatReactionSaveData.h"


XnatReactionSaveData::XnatReactionSaveData(QAction* parentObject, XnatUploadManager* uploadMngr, QWidget* p)
  : Superclass(parentObject), uploadManager(uploadMngr), parent(p)
{
    pqActiveObjects* activeObjects = &pqActiveObjects::instance();
    QObject::connect(activeObjects, SIGNAL(portChanged(pqOutputPort*)),
                     this, SLOT(updateEnableState()));
    this->updateEnableState();
}

void XnatReactionSaveData::updateEnableState()
{
    pqActiveObjects& activeObjects = pqActiveObjects::instance();
    // TODO: also is there's a pending accept.
    pqOutputPort* port = activeObjects.activePort();
    bool enable_state = ( port != NULL );
    if ( enable_state )
    {
        vtkSMWriterFactory* writerFactory =
            vtkSMProxyManager::GetProxyManager()->GetWriterFactory();
        enable_state = writerFactory->CanWrite(
            vtkSMSourceProxy::SafeDownCast(port->getSource()->getProxy()),
        port->getPortNumber());
    }
    this->parentAction()->setEnabled(enable_state);
}

void XnatReactionSaveData::saveSelectedData()
{
    pqServer* server = pqActiveObjects::instance().activeServer();
    // TODO: also is there's a pending accept.
    pqOutputPort* port = pqActiveObjects::instance().activePort();
    if ( !server || !port )
    {
        qCritical("No active source located.");
        return;
    }

    vtkSMWriterFactory* writerFactory =
        vtkSMProxyManager::GetProxyManager()->GetWriterFactory();
    QString filters = writerFactory->GetSupportedFileTypes(
        vtkSMSourceProxy::SafeDownCast(port->getSource()->getProxy()),
        port->getPortNumber());
    if ( filters.isEmpty() )
    {
        qCritical("Cannot determine writer to use.");
        return;
    }

    // print filter list
    // qCritical() << "filters: " << filters;

    // get name of file with valid type from user
    XnatFileNameTypeDialog nameTypeDialog(filters, parent);
    if ( nameTypeDialog.exec() != QDialog::Accepted )
    {
        return;
    }
    QString fname = nameTypeDialog.getFilename();

    // print filename input by user
    // qCritical() << "User input filename: " << fname;

    // get full path for subdirectory in work directory
    QString workSubdir = XnatBrowserSettings::getWorkSubdirectory();
    if ( workSubdir.isEmpty() )
    {
        return;
    }

    // construct full filename fname
    fname = QFileInfo(QDir(workSubdir), fname).absoluteFilePath();

    // print full filename
    // qCritical() << "Full path: " << fname;

    // save data to file(s) in sudirectory of work directory
    if ( !saveActiveData(fname) )
    {
        return;
    }

    // upload saved file(s) to XNAT
    uploadManager->uploadSavedData(workSubdir);
}

bool XnatReactionSaveData::saveActiveData(const QString& filename)
{
    pqServer* server = pqActiveObjects::instance().activeServer();
    // TODO: also is there's a pending accept.
    pqOutputPort* port = pqActiveObjects::instance().activePort();
    if ( !server || !port )
    {
        qCritical("No active source located.");
        return false;
    }

    vtkSMWriterFactory* writerFactory =
        vtkSMProxyManager::GetProxyManager()->GetWriterFactory();
    vtkSmartPointer<vtkSMProxy> proxy;
    proxy.TakeReference(writerFactory->CreateWriter(filename.toAscii().data(),
        vtkSMSourceProxy::SafeDownCast(port->getSource()->getProxy()),
        port->getPortNumber()));
    vtkSMSourceProxy* writer = vtkSMSourceProxy::SafeDownCast(proxy);
    if ( !writer )
    {
        qCritical() << "Failed to create writer for: " << filename;
        return false;
    }

    if ( writer->IsA("vtkSMPSWriterProxy") && port->getServer()->getNumberOfPartitions() > 1 )
    {
        //pqOptions* options = pqApplicationCore::instance()->getOptions();
        // To avoid showing the dialog when running tests.
        if ( !pqApplicationCore::instance()->testUtility()->playingTest() )
        {
            QMessageBox::StandardButton result = 
                QMessageBox::question(
                    pqCoreUtilities::mainWidget(),
                    "Serial Writer Warning",
                    "This writer will collect all of the data to the first node before "
                    "writing because it does not support parallel IO. This may cause the "
                    "first node to run out of memory if the data is large. "
                    "Are you sure you want to continue?",
                    QMessageBox::Ok | QMessageBox::Cancel,
                    QMessageBox::Cancel);
            if ( result == QMessageBox::Cancel )
            {
                return false;
            }
        }
    }

    pqWriterDialog dialog(writer);

    // Check to see if this writer has any properties that can be configured by 
    // the user. If it does, display the dialog.
    if ( dialog.hasConfigurableProperties() )
    {
        dialog.exec();
        if( dialog.result() == QDialog::Rejected )
        {
            // user pressed Cancel so don't write
            return false;
        }
    }
    writer->UpdateVTKObjects();
    writer->UpdatePipeline();
    return true;
}
