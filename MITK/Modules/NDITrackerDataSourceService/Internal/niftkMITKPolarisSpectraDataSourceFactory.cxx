/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKPolarisSpectraDataSourceFactory.h"
#include "niftkMITKTrackerDataSourceService.h"
#include <niftkSpectraTracker.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MITKPolarisSpectraDataSourceFactory::MITKPolarisSpectraDataSourceFactory()
: MITKTrackerDataSourceFactory("NDI Spectra")
{
}


//-----------------------------------------------------------------------------
MITKPolarisSpectraDataSourceFactory::~MITKPolarisSpectraDataSourceFactory()
{
}



//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer MITKPolarisSpectraDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{

  std::string portName;
  std::string fileName;

  this->ExtractProperties(properties, portName, fileName);

#ifdef __APPLE__
  portName = ConvertPortNameToPortIndex(portName);
#endif

  niftk::SpectraTracker::Pointer tracker = niftk::SpectraTracker::New(
        dataStorage, portName, fileName
        );

  niftk::MITKTrackerDataSourceService::Pointer serviceInstance
      = MITKTrackerDataSourceService::New(QString("PolarisSpectra"), // data source name
                                          this->GetName(),          // factory name
                                          properties,               // configure at startup
                                          dataStorage,
                                          tracker.GetPointer()
                                         );
  return serviceInstance.GetPointer();
}

} // end namespace
