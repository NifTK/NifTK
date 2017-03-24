/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtCameraVideoDataSourceService.h"
#include "cameraframegrabber.h"
#include <niftkQImageDataType.h>
#include <niftkQImageConversion.h>
#include <mitkExceptionMacro.h>
#include <QCamera>
#include <QCameraInfo>

namespace niftk
{

//-----------------------------------------------------------------------------
QtCameraVideoDataSourceService::QtCameraVideoDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: QImageDataSourceService(QString("QtVideo-"),
                          factoryName,
                          25, // expected frames per second (ignored, as Qt uses a signal).
                          50, // ring buffer size,
                          properties,
                          dataStorage
                         )
, m_Camera(nullptr)
, m_CameraFrameGrabber(nullptr)
{
  this->SetStatus("Initialising");

  if(!properties.contains("name"))
  {
    mitkThrow() << "Video device name not specified!";
  }
  QString videoDeviceName = (properties.value("name")).toString();

  QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
  foreach (const QCameraInfo &cameraInfo, cameras)
  {
    if (cameraInfo.deviceName() == videoDeviceName)
    {
      m_Camera = new QCamera(cameraInfo);
    }
  }
  if (m_Camera == nullptr)
  {
    mitkThrow() << "Failed to create video source:" << videoDeviceName.toStdString();
  }
  m_Camera->setCaptureMode(QCamera::CaptureVideo);

  m_CameraFrameGrabber = new CameraFrameGrabber();
  m_Camera->setViewfinder(m_CameraFrameGrabber);

  bool ok = QObject::connect(m_CameraFrameGrabber, SIGNAL(frameAvailable(QImage)),
                             this, SLOT(OnFrameAvailable(QImage)), Qt::DirectConnection);
  assert(ok);

  m_Camera->start();

  this->SetDescription("Local Qt video source.");
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
QtCameraVideoDataSourceService::~QtCameraVideoDataSourceService()
{
  if (m_Camera != nullptr)
  {
    m_Camera->stop();
    delete m_Camera;
  }
  if (m_CameraFrameGrabber != nullptr)
  {
    delete m_CameraFrameGrabber;
  }
}


//-----------------------------------------------------------------------------
std::unique_ptr<niftk::IGIDataType> QtCameraVideoDataSourceService::GrabImage()
{
  // So this method has to create a clone, to pass as a return
  // value, like a factory method would do.

  niftk::QImageDataType* wrapper = new niftk::QImageDataType();
  wrapper->SetImage(m_TemporaryWrapper); // clones it.

  std::unique_ptr<niftk::IGIDataType> result(wrapper);
  return result;
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataSourceService::OnFrameAvailable(const QImage &image)
{
  // Note: We can't take ownership of input image.

  m_TemporaryWrapper = new QImage(image); // should just wrap without copy.
  this->GrabData();
  delete m_TemporaryWrapper;              // then we delete this temporary wrapper.
}

} // end namespace
