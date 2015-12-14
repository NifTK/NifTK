/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKAuroraTableTopDataSourceFactory.h"
#include "niftkMITKTrackerDataSourceService.h"
#include <niftkAuroraTableTopTracker.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MITKAuroraTableTopDataSourceFactory::MITKAuroraTableTopDataSourceFactory()
: MITKTrackerDataSourceFactory("NDI Aurora (Table Top)")
{
}


//-----------------------------------------------------------------------------
MITKAuroraTableTopDataSourceFactory::~MITKAuroraTableTopDataSourceFactory()
{
}



//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer MITKAuroraTableTopDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{

  mitk::SerialCommunication::PortNumber portNumber;
  std::string fileName;

  this->ExtractProperties(properties, portNumber, fileName);

  niftk::AuroraTableTopTracker::Pointer tracker = niftk::AuroraTableTopTracker::New(
        dataStorage, portNumber, fileName
        );

  niftk::MITKTrackerDataSourceService::Pointer serviceInstance
      = MITKTrackerDataSourceService::New(QString("AuroraTableTop-"), // data source name
                                          this->GetName(),        // factory name
                                          properties,             // configure at startup
                                          dataStorage,
                                          tracker.GetPointer()
                                         );
  return serviceInstance.GetPointer();
}

} // end namespace
