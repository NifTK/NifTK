/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNVidiaSDIDataSourceFactory.h"
#include "niftkNVidiaSDIDataSourceService.h"
#include <niftkLagDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NVidiaSDIDataSourceFactory::NVidiaSDIDataSourceFactory()
: IGIDataSourceFactoryServiceI("NVIDIA SDI",
                               false, // don't need to configure at startup
                               true   // can configure while running
                               )
{
}


//-----------------------------------------------------------------------------
NVidiaSDIDataSourceFactory::~NVidiaSDIDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer NVidiaSDIDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{
  niftk::NVidiaSDIDataSourceService::Pointer serviceInstance
      = NVidiaSDIDataSourceService::New(this->GetName(), // factory name
                                        properties,      // configure at startup
                                        dataStorage
                                        );

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* NVidiaSDIDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  mitkThrow() << "NVidiaSDIDataSourceFactory does not provide an initialisation dialog.";
  return NULL;
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* NVidiaSDIDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> NVidiaSDIDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  return names;
}

} // end namespace
