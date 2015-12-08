/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyLinkClientDataSourceService.h"

namespace niftk
{

//-----------------------------------------------------------------------------
NiftyLinkClientDataSourceService::NiftyLinkClientDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage
    )
: NiftyLinkDataSourceService(QString("Matt, fixme"), factoryName, properties, dataStorage)
{

}


//-----------------------------------------------------------------------------
NiftyLinkClientDataSourceService::~NiftyLinkClientDataSourceService()
{

}

} // end namespace
