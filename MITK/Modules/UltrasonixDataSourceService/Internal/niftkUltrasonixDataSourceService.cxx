/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUltrasonixDataSourceService.h"
#include <niftkQImageDataType.h>
#include <niftkQImageConversion.h>
#include <mitkExceptionMacro.h>
#include <QtGui/QtGui>

#define BUFFERSIZE (4*1024*1024)

namespace niftk
{
UltrasonixDataSourceInterface* UltrasonixDataSourceInterface::s_Instance = nullptr;

// nice way to implement sleep since Qt typically requires setup of threads to use usleep(), etc.
void QSleep(int time)
{
  QMutex mtx;
  mtx.lock();
  QWaitCondition wc;
  wc.wait(&mtx, time);
}

//-----------------------------------------------------------------------------
UltrasonixDataSourceInterface* UltrasonixDataSourceInterface::CreateInstance(UltrasonixDataSourceService* serviceObj)
{
  if (s_Instance == nullptr)
  {
    s_Instance = new UltrasonixDataSourceInterface(serviceObj);
  }
  else
  {
    mitkThrow() << "You cannot create more than 1 Ultrasonix Interface.";
  }

  return s_Instance;
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceInterface* UltrasonixDataSourceInterface::GetInstance()
{
  if (s_Instance == nullptr)
  {
    mitkThrow() << "Instance should already be created.";
  }
  return s_Instance;
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceInterface::UltrasonixDataSourceInterface(UltrasonixDataSourceService* serviceObj)
: m_Ulterius(nullptr)
, m_Service(nullptr)
, m_Buffer(nullptr)
{
  if (serviceObj == nullptr)
  {
    mitkThrow() << "The service is null.";
  }

  m_Buffer = new unsigned char[BUFFERSIZE];

  m_Service = serviceObj;
  m_Ulterius = new ulterius;
  m_Ulterius->setCallback(NewDataCallBack);
  m_Ulterius->setParamCallback(ParamCallBack);
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceInterface::~UltrasonixDataSourceInterface()
{
  this->Disconnect();

  m_Service = nullptr;
  delete m_Ulterius;
  delete [] m_Buffer;
  s_Instance = nullptr;
}


//-----------------------------------------------------------------------------
bool UltrasonixDataSourceInterface::ProcessBuffer(void *data, int type, int sz, bool cine, int frmnum)
{
  uDataDesc desc;
  m_Ulterius->getDataDescriptor(static_cast<uData>(type), desc);

  if (desc.ss == 32)
  {
    // ulterius reports bogus alpha channel, in case of 32-bit images.
    unsigned int numberOfPixelsInImage = desc.w * desc.h;
    unsigned char* readFrom = static_cast<unsigned char*>(data);
    unsigned char* writeTo = m_Buffer;

    for (unsigned int i = 0; i < numberOfPixelsInImage; ++i)
    {
      // This should set the alpha channel to 1, leaving other pixels (RGB) unchanged.
      // use with QImage::Format_ARGB32
      ((unsigned int*)m_Buffer)[i] = static_cast<unsigned int*>(data)[i] | 0xFF000000;
    }
    QImage image(m_Buffer, desc.w, desc.h, QImage::Format_ARGB32);
    m_Service->ProcessImage(image);
  }
  else if (desc.ss == 8)
  {
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
    QImage image(static_cast<unsigned char*>(data), desc.w, desc.h, QImage::Format_Grayscale8);
#else
    QImage image(static_cast<unsigned char*>(data), desc.w, desc.h, QImage::Format_Indexed8);
#endif
    m_Service->ProcessImage(image);
  }
  else
  {
    MITK_WARN << "UltrasonixDataSourceInterface: Cannot process images with " << desc.ss << " bits.";
  }
  return true;
}


//-----------------------------------------------------------------------------
bool UltrasonixDataSourceInterface::ProcessParameterChange(void* paramID, int x, int y)
{
  char* paramName = static_cast<char*>(paramID);
  MITK_INFO << "Param updated:" << paramName << ", " << x << ", " << y;
  return true;
}


//-----------------------------------------------------------------------------
bool UltrasonixDataSourceInterface::NewDataCallBack(void *data, int type, int sz, bool cine, int frmnum)
{
  UltrasonixDataSourceInterface* us = UltrasonixDataSourceInterface::GetInstance();
  return us->ProcessBuffer(data, type, sz, cine, frmnum);
}


//-----------------------------------------------------------------------------
bool UltrasonixDataSourceInterface::ParamCallBack(void* paramID, int x, int y)
{
  UltrasonixDataSourceInterface* us = UltrasonixDataSourceInterface::GetInstance();
  return us->ProcessParameterChange(paramID, x, y);
}


//-----------------------------------------------------------------------------
bool UltrasonixDataSourceInterface::IsConnected() const
{
  return (m_Ulterius != nullptr && m_Ulterius->isConnected());
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceInterface::Connect(const QString& host)
{
  if (!m_Ulterius->connect(host.toAscii()))
  {
    mitkThrow() << "Failed to connect to:" << host.toStdString();
  }

  if ( m_Ulterius->getFreezeState() )  // 1: The system is frozen and not imaging. 0: The system is imaging.
  {
    m_Ulterius->toggleFreeze();
  }
  m_Ulterius->setDataToAcquire(udtBPost);
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceInterface::Disconnect()
{
  if (this->IsConnected())
  {
    m_Ulterius->setDataToAcquire(0);
    QSleep(500);
    if (!m_Ulterius->disconnect())
    {
      mitkThrow() << "Failed to disconnect!";
    }
    QSleep(500);
  }
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceService::UltrasonixDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: QImageDataSourceService(QString("Ultrasonix-"),
                          factoryName,
                          40, // expected frames per second (ignored, as SDK uses a callback).
                          80, // ring buffer size
                          properties,
                          dataStorage
                         )
, m_Ultrasonix(nullptr)
{
  this->SetStatus("Initialising");

  UltrasonixDataSourceInterface* ultrasonix = UltrasonixDataSourceInterface::CreateInstance(this);
  if (ultrasonix == nullptr)
  {
    mitkThrow() << "Failed to instantiate Ultrasonix Interface.";
  }
  m_Ultrasonix = ultrasonix;

  if(!properties.contains("host"))
  {
    mitkThrow() << "Host name not specified!";
  }
  QString host = (properties.value("host")).toString();

  // Basically, the data source should connect, stay connected and continuously stream.
  // If the Sonix MDP is 'frozen' then assumedly the callback is not called, but the
  // connection remains live.

  m_Ultrasonix->Connect(host);

  this->SetDescription("Ultrasonix Ulterius source.");
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceService::~UltrasonixDataSourceService()
{
  if (m_Ultrasonix->IsConnected())
  {
    m_Ultrasonix->Disconnect();
  }
  delete m_Ultrasonix;
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceService::ProcessImage(const QImage& image)
{
  // Note: We can't take ownership of input image.

  m_TemporaryImage = new QImage(image); // should just wrap without copy.
  this->GrabData();
  delete m_TemporaryImage;              // then we delete this temporary wrapper.
}


//-----------------------------------------------------------------------------
std::unique_ptr<niftk::IGIDataType> UltrasonixDataSourceService::GrabImage()
{
  // So this method has to create a clone, to pass as a return
  // value, like a factory method would do.

  niftk::QImageDataType* wrapper = new niftk::QImageDataType();
  wrapper->SetImage(m_TemporaryImage); // clones it.

  std::unique_ptr<niftk::IGIDataType> result(wrapper);
  return result;
}

//-----------------------------------------------------------------------------
void UltrasonixDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  niftk::SingleFrameDataSourceService::SetProperties(properties);


}


//-----------------------------------------------------------------------------
IGIDataSourceProperties UltrasonixDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props = niftk::SingleFrameDataSourceService::GetProperties();

  return props;
}


} // end namespace
