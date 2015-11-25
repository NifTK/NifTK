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
: IGIDataSourceFactoryServiceI("OpenCVVideoDataSourceFactory",
                               "OpenCVVideoDataSourceService",
                               "OpenCVVideoDataSourceServiceGui"
                               )
{
}


//-----------------------------------------------------------------------------
OpenCVVideoDataSourceFactory::~OpenCVVideoDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceServiceI* OpenCVVideoDataSourceFactory::Create(mitk::DataStorage::Pointer dataStorage)
{
  IGIDataSourceServiceI* serviceInstance = OpenCVVideoDataSourceService::New(dataStorage);
  return serviceInstance;
}

} // end namespace
