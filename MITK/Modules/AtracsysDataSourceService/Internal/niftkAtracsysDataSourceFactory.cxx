/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAtracsysDataSourceFactory.h"
#include "niftkAtracsysDataSourceService.h"
#include <niftkConfigFileDialog.h>
#include <niftkLagDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
AtracsysDataSourceFactory::AtracsysDataSourceFactory()
: IGIDataSourceFactoryServiceI("Atracsys",
                               true,  // configure IGTToolStorage file name.
                               true   // can configure lag while running
                               )
{
}


//-----------------------------------------------------------------------------
AtracsysDataSourceFactory::~AtracsysDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer AtracsysDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{
  niftk::AtracsysDataSourceService::Pointer serviceInstance
      = AtracsysDataSourceService::New(this->GetName(), // factory name
                                       properties,      // configure at startup
                                       dataStorage
                                      );

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* AtracsysDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  return new niftk::ConfigFileDialog(parent,
                                     "AtracsysDataSourceService",
                                     "uk.ac.ucl.cmic.niftkAtracsysDataSourceService.ConfigFileDialog");
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* AtracsysDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> AtracsysDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  return names;
}

} // end namespace
