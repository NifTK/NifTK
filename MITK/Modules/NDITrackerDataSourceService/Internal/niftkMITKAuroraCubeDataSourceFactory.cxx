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
#include "niftkMITKTrackerDialog.h"
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
IGIInitialisationDialog* MITKAuroraCubeDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  return new niftk::MITKTrackerDialog(parent, this->GetName(), 115200);
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer MITKAuroraCubeDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{

  std::string portName;
  std::string fileName;
  int         baudRate;

  this->ExtractProperties(properties, portName, fileName, baudRate);

  niftk::AuroraCubeTracker::Pointer tracker = niftk::AuroraCubeTracker::New(
        dataStorage, portName, fileName, baudRate
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
