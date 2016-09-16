/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtCameraVideoDataSourceFactory.h"
#include "niftkQtCameraVideoDataSourceService.h"
#include "niftkQtCameraDialog.h"
#include <niftkLagDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
QtCameraVideoDataSourceFactory::QtCameraVideoDataSourceFactory()
: IGIDataSourceFactoryServiceI("Qt Video",
                               true, // can pick which camera source and save format at startup
                               true  // can configure lag while running
                               )
{
}


//-----------------------------------------------------------------------------
QtCameraVideoDataSourceFactory::~QtCameraVideoDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer QtCameraVideoDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{
  niftk::QtCameraVideoDataSourceService::Pointer serviceInstance
      = QtCameraVideoDataSourceService::New(this->GetName(), // factory name
                                          properties,      // configure at startup
                                          dataStorage
                                          );

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* QtCameraVideoDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  return new niftk::QtCameraDialog(parent);
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* QtCameraVideoDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> QtCameraVideoDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  names.push_back("QmitkIGIQtCameraDataSource");
  return names;
}

} // end namespace
