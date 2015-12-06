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
#include <niftkLagDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
OpenCVVideoDataSourceFactory::OpenCVVideoDataSourceFactory()
: IGIDataSourceFactoryServiceI("OpenCV Frame Grabber",
                               false, // don't need to configure at startup
                               true   // can configure lag while running
                               )
{
}


//-----------------------------------------------------------------------------
OpenCVVideoDataSourceFactory::~OpenCVVideoDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer OpenCVVideoDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{
  niftk::OpenCVVideoDataSourceService::Pointer serviceInstance
      = OpenCVVideoDataSourceService::New(this->GetName(), // factory name
                                          properties,      // configure at startup
                                          dataStorage
                                          );

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* OpenCVVideoDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  mitkThrow() << "OpenCVVideoDataSourceService does not need a configuration dialog.";
  return NULL;
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* OpenCVVideoDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> OpenCVVideoDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  names.push_back("QmitkIGIOpenCVDataSource");
  return names;
}

} // end namespace
