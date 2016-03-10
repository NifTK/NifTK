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
#include "niftkNVidiaSDIInitDialog.h"
#include "niftkNVidiaSDIConfigDialog.h"

namespace niftk
{

//-----------------------------------------------------------------------------
NVidiaSDIDataSourceFactory::NVidiaSDIDataSourceFactory()
: IGIDataSourceFactoryServiceI("NVIDIA SDI",
                               true,  // configure the type at startup
                               true   // can configure lag while running
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
  return new niftk::NVidiaSDIInitDialog(parent);
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* NVidiaSDIDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::NVidiaSDIConfigDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> NVidiaSDIDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  names.push_back("QmitkIGINVidiaDataSource");
  return names;
}

} // end namespace
