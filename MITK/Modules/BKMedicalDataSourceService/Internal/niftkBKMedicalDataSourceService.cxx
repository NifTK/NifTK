/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBKMedicalDataSourceService.h"
#include <niftkQImageDataType.h>
#include <niftkQImageConversion.h>
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
BKMedicalDataSourceService::BKMedicalDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: QImageDataSourceService(QString("BKMedical-"),
                          factoryName,
                          40, // expected frames per second,
                          80, // ring buffer size,
                          properties,
                          dataStorage
                         )
{
  if(!properties.contains("host"))
  {
    mitkThrow() << "Host name not specified!";
  }
  QString host = (properties.value("host")).toString();

  this->SetStatus("Initialising");

  int portNumber = 7915;

  m_Worker = new BKMedicalDataSourceWorker();
  m_Worker->ConnectToHost(host, portNumber); // must throw if failed.
  m_Worker->moveToThread(&m_WorkerThread);

  connect(m_Worker, SIGNAL(ImageReceived(QImage)), this, SLOT(OnFrameAvailable(QImage)), Qt::DirectConnection);
  connect(&m_WorkerThread, SIGNAL(finished()), m_Worker, SLOT(deleteLater()));
  connect(&m_WorkerThread, SIGNAL(started()), m_Worker, SLOT(ReceiveImages()));

  m_WorkerThread.start();

  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
BKMedicalDataSourceService::~BKMedicalDataSourceService()
{
  if (m_WorkerThread.isRunning())
  {
    m_WorkerThread.quit();
    m_WorkerThread.wait();
  }
}


//-----------------------------------------------------------------------------
std::unique_ptr<niftk::IGIDataType> BKMedicalDataSourceService::GrabImage()
{
  // So this method has to create a clone, to pass as a return
  // value, like a factory method would do.

  niftk::QImageDataType* wrapper = new niftk::QImageDataType();
  wrapper->SetImage(m_TemporaryWrapper); // clones it.

  std::unique_ptr<niftk::IGIDataType> result(wrapper);
  return result;
}


//-----------------------------------------------------------------------------
void BKMedicalDataSourceService::OnFrameAvailable(const QImage& image)
{
  // Note: We can't take ownership of input image.
  m_TemporaryWrapper = new QImage(image); // should just wrap without deep copy.
  this->GrabData();
  delete m_TemporaryWrapper;              // then we delete this temporary wrapper.
}

} // end namespace
