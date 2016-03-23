/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyLinkClientDataSourceFactory.h"
#include "niftkNiftyLinkClientDataSourceService.h"
#include <niftkIPHostPortDialog.h>
#include <niftkLagDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NiftyLinkClientDataSourceFactory::NiftyLinkClientDataSourceFactory()
: IGIDataSourceFactoryServiceI("OpenIGTLink Client",
                               true, // can configure hostname, port number
                               true  // can configure lag while running
                               )
{
}


//-----------------------------------------------------------------------------
NiftyLinkClientDataSourceFactory::~NiftyLinkClientDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* NiftyLinkClientDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  return new niftk::IPHostPortDialog(parent);
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* NiftyLinkClientDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> NiftyLinkClientDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  return names;
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer NiftyLinkClientDataSourceFactory::CreateService(mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{
  niftk::NiftyLinkClientDataSourceService::Pointer serviceInstance
      = NiftyLinkClientDataSourceService::New(this->GetName(), // factory name
                                              properties,      // configure at startup
                                              dataStorage
                                             );

  return serviceInstance.GetPointer();
}

} // end namespace
