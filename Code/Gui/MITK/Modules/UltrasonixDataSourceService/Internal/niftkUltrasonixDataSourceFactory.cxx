/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUltrasonixDataSourceFactory.h"
#include "niftkUltrasonixDataSourceService.h"
#include <niftkLagDialog.h>
#include <niftkIPHostPortDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
UltrasonixDataSourceFactory::UltrasonixDataSourceFactory()
: IGIDataSourceFactoryServiceI("Ultrasonix",
                               true,  // configure host and port at startup
                               true   // can configure lag while running
                               )
{
}


//-----------------------------------------------------------------------------
UltrasonixDataSourceFactory::~UltrasonixDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer UltrasonixDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{
  niftk::UltrasonixDataSourceService::Pointer serviceInstance
      = UltrasonixDataSourceService::New(this->GetName(), // factory name
                                          properties,      // configure at startup
                                          dataStorage
                                          );

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* UltrasonixDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  return new niftk::IPHostPortDialog(parent);
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* UltrasonixDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> UltrasonixDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  return names;
}

} // end namespace
