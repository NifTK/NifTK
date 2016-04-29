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
#include "niftkMITKTrackerDialog.h"
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
IGIInitialisationDialog* MITKPolarisSpectraDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  return new niftk::MITKTrackerDialog(parent, this->GetName(), 1228739);
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer MITKPolarisSpectraDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{

  std::string portName;
  std::string fileName;
  int         baudRate;

  this->ExtractProperties(properties, portName, fileName, baudRate);

  niftk::SpectraTracker::Pointer tracker = niftk::SpectraTracker::New(
        dataStorage, portName, fileName, baudRate
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
