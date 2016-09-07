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

#define BUFFERSIZE (4*1024*1024)

namespace niftk
{
UltrasonixDataSourceInterface* UltrasonixDataSourceInterface::s_Instance = nullptr;

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
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceInterface::ProcessBuffer(void *data, int type, int sz, bool cine, int frmnum)
{
  uDataDesc desc;
  m_Ulterius->getDataDescriptor(static_cast<uData>(type), desc);

  // all other image types are either fine or untested.
  if (desc.type == 0x00000008) // udtBPost32
  {
    // ulterius reports bogus alpha channel, in case of 32-bit images.
    unsigned int numberOfPixelsInImage = desc.w * desc.h;
    for (unsigned int i = 0; i < numberOfPixelsInImage; ++i)
    {
      // the channel layout coming from ulterius is in the correct order for QImage::Format_ARGB32,
      // this is known as BGRA elsewhere, i.e. red = ((unsigned char*) &((unsigned int*) buffer)[x, y])[2];
      m_Buffer[i] = static_cast<unsigned int*>(data)[i] | 0xFF000000;
    }
    QImage image(m_Buffer, desc.w, desc.h, QImage::Format_ARGB32);
    m_Service->ProcessImage(image);
  }
  else
  {
    QImage image(static_cast<unsigned char*>(data), desc.w, desc.h, QImage::Format_Indexed8);
    m_Service->ProcessImage(image);
  }

}


//-----------------------------------------------------------------------------
bool UltrasonixDataSourceInterface::NewDataCallBack(void *data, int type, int sz, bool cine, int frmnum)
{
  UltrasonixDataSourceInterface* us = UltrasonixDataSourceInterface::GetInstance();
  us->ProcessBuffer(data, type, sz, cine, frmnum);
  return true;
}


//-----------------------------------------------------------------------------
bool UltrasonixDataSourceInterface::ParamCallBack(void* paramID, int x, int y)
{
  MITK_INFO << "Param updated";
  return true;
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
}


//-----------------------------------------------------------------------------
void UltrasonixDataSourceInterface::Disconnect()
{
  if (this->IsConnected())
  {
    if (!m_Ulterius->disconnect())
    {
      mitkThrow() << "Failed to disconnect!";
    }
  }
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceService::UltrasonixDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: QImageDataSourceService(QString("Ultrasonix-"), factoryName, properties, dataStorage)
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

  if(!properties.contains("extension"))
  {
    mitkThrow() << "File extension not specified!";
  }
  QString extension = (properties.value("extension")).toString();

  mitkThrow() << "Not implemented yet. Volunteers .... please step forward!";

  // Basically, the data source should connect, stay connected and continuously stream.
  // If the Sonix MDP is 'frozen' then assumedly the callback is not called, but the 
  // connection remains live.

  m_Ultrasonix->Connect(host);

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
  m_TemporaryImage = new QImage(image); // should just wrap without copy.
  this->GrabData();
  delete m_TemporaryImage;              // then we delete this temporary wrapper.
}


//-----------------------------------------------------------------------------
niftk::IGIDataType::Pointer UltrasonixDataSourceService::GrabImage()
{
  niftk::QImageDataType::Pointer image = niftk::QImageDataType::New();
  image->DeepCopy(*m_TemporaryImage);  // this should be the only copy in the pipeline.
  return image.GetPointer();
}

} // end namespace
