/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOpenCVVideoDataSourceFactory.h"
#include "niftkOpenCVVideoDataSourceService.h"

namespace niftk
{

//-----------------------------------------------------------------------------
OpenCVVideoDataSourceFactory::OpenCVVideoDataSourceFactory()
: IGIDataSourceFactoryServiceI("OpenCV Frame Grabber",
                               "OpenCVVideoDataSourceService",
                               false,
                               "", // don't need a startup GUI, nothing to configure
                               ""  // don't need an observation GUI, no parameters to tweak
                               )
{
}


//-----------------------------------------------------------------------------
OpenCVVideoDataSourceFactory::~OpenCVVideoDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer OpenCVVideoDataSourceFactory::Create(
    mitk::DataStorage::Pointer dataStorage)
{
  niftk::OpenCVVideoDataSourceService::Pointer serviceInstance
      = OpenCVVideoDataSourceService::New(this->GetName(), // factory name
                                          dataStorage);

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer OpenCVVideoDataSourceFactory::Create(
    const std::string& name,
    mitk::DataStorage::Pointer dataStorage)
{
  niftk::OpenCVVideoDataSourceService::Pointer serviceInstance
      = OpenCVVideoDataSourceService::New(name,
                                          this->GetName(), // factory name
                                          dataStorage);

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
std::vector<std::string> OpenCVVideoDataSourceFactory::GetLegacyClassNames() const
{
  std::vector<std::string> names;
  names.push_back("QmitkIGIOpenCVDataSource");
  return names;
}

} // end namespace
