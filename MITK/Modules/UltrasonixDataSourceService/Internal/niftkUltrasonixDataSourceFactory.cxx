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
#include "niftkUltrasonixConfigDialog.h"
#include <niftkIPHostPortExtensionDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
UltrasonixDataSourceFactory::UltrasonixDataSourceFactory()
: IGIDataSourceFactoryServiceI("Ultrasonix (Local)",
                               true,  // configure host, port, extension at startup
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
  QStringList names;
  names.append("JPEG");
  names.append("PNG");

  QStringList extensions;
  extensions.append(".jpg");
  extensions.append(".png");

  QString settings("uk.ac.ucl.cmic.niftkUltrasonixDataSourceFactory.IPHostPortExtensionDialog");

  niftk::IPHostPortExtensionDialog* dialog = new niftk::IPHostPortExtensionDialog(parent, settings, names, extensions);
  dialog->SetHostVisible(false);
  dialog->SetPortVisible(true);
  dialog->SetExtensionVisible(true);

  return dialog;
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* UltrasonixDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::UltrasonixConfigDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> UltrasonixDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  return names;
}

} // end namespace
