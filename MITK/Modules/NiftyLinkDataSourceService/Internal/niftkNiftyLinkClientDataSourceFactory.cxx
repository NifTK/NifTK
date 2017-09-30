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
#include <niftkIPHostPortExtensionDialog.h>
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
  QStringList names;
  names.append(".nii");
  names.append(".nii.gz");
  names.append(".jpg");
  names.append(".png");

  QStringList extensions;
  extensions.append(".nii");
  extensions.append(".nii.gz");
  extensions.append(".jpg");
  extensions.append(".png");

  QString settings("uk.ac.ucl.cmic.niftkNiftyLinkClientDataSourceFactory.IPHostPortExtensionDialog");

  niftk::IPHostPortExtensionDialog* dialog = new niftk::IPHostPortExtensionDialog(parent,
                                                                                  settings,
                                                                                  3200,
                                                                                  names,
                                                                                  extensions);
  dialog->SetHostVisible(true);
  dialog->SetPortVisible(true);
  dialog->SetExtensionVisible(true);

  return dialog;
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
