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

//-----------------------------------------------------------------------------
UltrasonixDataSourceService::UltrasonixDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: QImageDataSourceService(QString("Ultrasonix-"), factoryName, properties, dataStorage)
{
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

  this->SetStatus("Initialising");

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
}

} // end namespace
