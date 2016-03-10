/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKPolarisVicraDataSourceFactory.h"
#include "niftkMITKTrackerDataSourceService.h"
#include <niftkVicraTracker.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MITKPolarisVicraDataSourceFactory::MITKPolarisVicraDataSourceFactory()
: MITKTrackerDataSourceFactory("NDI Vicra")
{
}


//-----------------------------------------------------------------------------
MITKPolarisVicraDataSourceFactory::~MITKPolarisVicraDataSourceFactory()
{
}



//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer MITKPolarisVicraDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{

  mitk::SerialCommunication::PortNumber portNumber;
  std::string fileName;

  this->ExtractProperties(properties, portNumber, fileName);

  niftk::VicraTracker::Pointer tracker = niftk::VicraTracker::New(
        dataStorage, portNumber, fileName
        );

  niftk::MITKTrackerDataSourceService::Pointer serviceInstance
      = MITKTrackerDataSourceService::New(QString("PolarisVicra"), // data source name
                                          this->GetName(),          // factory name
                                          properties,               // configure at startup
                                          dataStorage,
                                          tracker.GetPointer()
                                         );
  return serviceInstance.GetPointer();
}

} // end namespace
