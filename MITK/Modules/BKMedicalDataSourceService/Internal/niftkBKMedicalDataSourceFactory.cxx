/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBKMedicalDataSourceFactory.h"
#include "niftkBKMedicalDataSourceService.h"
#include <niftkLagDialog.h>
#include <niftkIPHostExtensionDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
BKMedicalDataSourceFactory::BKMedicalDataSourceFactory()
: IGIDataSourceFactoryServiceI("BKMedical (Local)",
                               true,  // configure host, extension at startup
                               true   // can configure lag while running
                               )
{
}


//-----------------------------------------------------------------------------
BKMedicalDataSourceFactory::~BKMedicalDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer BKMedicalDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{
  niftk::BKMedicalDataSourceService::Pointer serviceInstance
      = BKMedicalDataSourceService::New(this->GetName(), // factory name
                                         properties,      // configure at startup
                                         dataStorage
                                        );

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* BKMedicalDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  return new niftk::IPHostExtensionDialog(parent,
                                          "uk.ac.ucl.cmic.niftkBKMedicalDataSourceService.IPHostExtensionDialog");
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* BKMedicalDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> BKMedicalDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  return names;
}

} // end namespace
