/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKAuroraCubeDataSourceFactory.h"
#include "niftkMITKTrackerDataSourceService.h"
#include <niftkAuroraCubeTracker.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MITKAuroraCubeDataSourceFactory::MITKAuroraCubeDataSourceFactory()
: MITKTrackerDataSourceFactory("NDI Aurora (Cube)")
{
}


//-----------------------------------------------------------------------------
MITKAuroraCubeDataSourceFactory::~MITKAuroraCubeDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer MITKAuroraCubeDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{

  std::string portName;
  std::string fileName;

  this->ExtractProperties(properties, portName, fileName);

  niftk::AuroraCubeTracker::Pointer tracker = niftk::AuroraCubeTracker::New(
        dataStorage, portName, fileName
        );

  niftk::MITKTrackerDataSourceService::Pointer serviceInstance
      = MITKTrackerDataSourceService::New(QString("AuroraCube"), // data source name
                                          this->GetName(),        // factory name
                                          properties,             // configure at startup
                                          dataStorage,
                                          tracker.GetPointer()
                                         );
  return serviceInstance.GetPointer();
}

} // end namespace
