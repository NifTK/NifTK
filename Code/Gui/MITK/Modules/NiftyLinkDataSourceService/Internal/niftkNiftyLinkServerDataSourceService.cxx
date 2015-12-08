/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyLinkServerDataSourceService.h"

namespace niftk
{

//-----------------------------------------------------------------------------
NiftyLinkServerDataSourceService::NiftyLinkServerDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage
    )
: NiftyLinkDataSourceService(QString("Matt, fixme"), factoryName, properties, dataStorage)
{

}


//-----------------------------------------------------------------------------
NiftyLinkServerDataSourceService::~NiftyLinkServerDataSourceService()
{

}

} // end namespace
