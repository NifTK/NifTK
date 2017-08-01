/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyLinkServerDataSourceFactory.h"
#include "niftkNiftyLinkServerDataSourceService.h"
#include <niftkIPHostPortExtensionDialog.h>
#include <niftkLagDialog.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NiftyLinkServerDataSourceFactory::NiftyLinkServerDataSourceFactory()
: IGIDataSourceFactoryServiceI("OpenIGTLink Server",
                               true, // can configure port number
                               true  // can configure lag while running
                               )
{
}


//-----------------------------------------------------------------------------
NiftyLinkServerDataSourceFactory::~NiftyLinkServerDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* NiftyLinkServerDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  QStringList names;
  names.append("NIfTI");
  names.append("NIfTI");
  names.append("JPEG");
  names.append("PNG");

  QStringList extensions;
  extensions.append(".nii");
  extensions.append(".nii.gz");
  extensions.append(".jpg");
  extensions.append(".png");

  QString settings("uk.ac.ucl.cmic.niftkNiftyLinkClientDataSourceFactory.IPHostPortExtensionDialog");

  niftk::IPHostPortExtensionDialog* dialog = new niftk::IPHostPortExtensionDialog(parent, settings, 3200, names, extensions);
  dialog->SetHostVisible(false);
  dialog->SetPortVisible(true);
  dialog->SetExtensionVisible(true);

  return dialog;
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* NiftyLinkServerDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  return new niftk::LagDialog(parent, service);
}


//-----------------------------------------------------------------------------
QList<QString> NiftyLinkServerDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  return names;
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer NiftyLinkServerDataSourceFactory::CreateService(mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{
  niftk::NiftyLinkServerDataSourceService::Pointer serviceInstance
      = NiftyLinkServerDataSourceService::New(this->GetName(), // factory name
                                              properties,      // configure at startup
                                              dataStorage
                                             );

  return serviceInstance.GetPointer();
}

} // end namespace
