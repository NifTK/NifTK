/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKAuroraDomeDataSourceFactory.h"
#include "niftkMITKTrackerDataSourceService.h"
#include "niftkMITKTrackerDialog.h"
#include <niftkAuroraDomeTracker.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MITKAuroraDomeDataSourceFactory::MITKAuroraDomeDataSourceFactory()
: MITKTrackerDataSourceFactory("NDI Aurora (Dome)")
{
}


//-----------------------------------------------------------------------------
MITKAuroraDomeDataSourceFactory::~MITKAuroraDomeDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* MITKAuroraDomeDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  return new niftk::MITKTrackerDialog(parent, this->GetName(), 115200);
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer MITKAuroraDomeDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{

  std::string portName;
  std::string fileName;
  int         baudRate;

  this->ExtractProperties(properties, portName, fileName, baudRate);

  niftk::AuroraDomeTracker::Pointer tracker = niftk::AuroraDomeTracker::New(
        dataStorage, portName, fileName, baudRate
        );

  niftk::MITKTrackerDataSourceService::Pointer serviceInstance
      = MITKTrackerDataSourceService::New(QString("AuroraDome"), // data source name
                                          this->GetName(),        // factory name
                                          properties,             // configure at startup
                                          dataStorage,
                                          tracker.GetPointer()
                                         );
  return serviceInstance.GetPointer();
}

} // end namespace
