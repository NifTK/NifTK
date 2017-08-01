/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBlackMagicDataSourceFactory.h"
#include "niftkBlackMagicDataSourceService.h"
#include <niftkLagDialog.h>
#include <niftkIPHostPortExtensionDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
BlackMagicDataSourceFactory::BlackMagicDataSourceFactory()
: IGIDataSourceFactoryServiceI("BlackMagic",
                               true,  // configure whatever at startup
                               true   // can configure lag while running
                               )
{
}


//-----------------------------------------------------------------------------
BlackMagicDataSourceFactory::~BlackMagicDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer BlackMagicDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{
  niftk::BlackMagicDataSourceService::Pointer serviceInstance
      = BlackMagicDataSourceService::New(this->GetName(), // factory name
                                          properties,      // configure at startup
                                          dataStorage
                                          );

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* BlackMagicDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  // Return a dialog box to configure things at startup.
  // This one here is provided as an example only.
  QStringList names;
  names.append("JPEG");
  names.append("PNG");

  QStringList extensions;
  extensions.append(".jpg");
  extensions.append(".png");

  QString settings("uk.ac.ucl.cmic.niftkBlackMagicDataSourceFactory.IPHostPortExtensionDialog");

  niftk::IPHostPortExtensionDialog* dialog = new niftk::IPHostPortExtensionDialog(parent, settings, names, extensions);
  dialog->SetHostVisible(false);
  dialog->SetPortVisible(true);
  dialog->SetExtensionVisible(true);

  return dialog;
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* BlackMagicDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> BlackMagicDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  return names;
}

} // end namespace
