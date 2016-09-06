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

namespace niftk
{
UltrasonixDataSourceInterface* UltrasonixDataSourceInterface::s_Instance = nullptr;

//-----------------------------------------------------------------------------
UltrasonixDataSourceInterface* UltrasonixDataSourceInterface::GetInstance()
{
  if (s_Instance == nullptr)
  {
    s_Instance = new UltrasonixDataSourceInterface();
  }
  else
  {
    mitkThrow() << "You cannot create >1 Ultrasonix Interface.";
  }

  return s_Instance;
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceInterface::UltrasonixDataSourceInterface()
: m_Ulterius(nullptr)
{
  m_Ulterius = new ulterius;
  m_Ulterius->setCallback(NewDataCallBack);
  m_Ulterius->setParamCallback(ParamCallBack);
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceInterface::~UltrasonixDataSourceInterface()
{
}


//-----------------------------------------------------------------------------
bool UltrasonixDataSourceInterface::NewDataCallBack(void *data, int type, int sz, bool cine, int frmnum)
{
  MITK_INFO << "New Data";
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
UltrasonixDataSourceService::UltrasonixDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: QImageDataSourceService(QString("Ultrasonix-"), factoryName, properties, dataStorage)
, m_Ultrasonix(nullptr)
{
  this->SetStatus("Initialising");

  UltrasonixDataSourceInterface* ultrasonix = UltrasonixDataSourceInterface::GetInstance();
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

  if(!properties.contains("port"))
  {
    mitkThrow() << "Port number not specified!";
  }
  int portNumber = properties.value("port").toInt();

  mitkThrow() << "Not implemented yet. Volunteers .... please step forward!";

  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceService::~UltrasonixDataSourceService()
{
}


//-----------------------------------------------------------------------------
niftk::IGIDataType::Pointer UltrasonixDataSourceService::GrabImage()
{
  niftk::IGIDataType::Pointer result;
  return result;
}

} // end namespace
