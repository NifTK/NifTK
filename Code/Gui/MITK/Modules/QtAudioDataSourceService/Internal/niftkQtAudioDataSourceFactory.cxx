/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtAudioDataSourceFactory.h"
#include "niftkQtAudioDataSourceService.h"

namespace niftk
{

//-----------------------------------------------------------------------------
QtAudioDataSourceFactory::QtAudioDataSourceFactory()
: IGIDataSourceFactoryServiceI("Audio",
                               false,  // configure host and port at startup
                               false   // can configure lag while running
                               )
{
}


//-----------------------------------------------------------------------------
QtAudioDataSourceFactory::~QtAudioDataSourceFactory()
{
}


//-----------------------------------------------------------------------------
IGIDataSourceI::Pointer QtAudioDataSourceFactory::CreateService(
    mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const
{
  niftk::QtAudioDataSourceService::Pointer serviceInstance
      = QtAudioDataSourceService::New(this->GetName(), // factory name
                                          properties,      // configure at startup
                                          dataStorage
                                          );

  return serviceInstance.GetPointer();
}


//-----------------------------------------------------------------------------
IGIInitialisationDialog* QtAudioDataSourceFactory::CreateInitialisationDialog(QWidget *parent) const
{
  mitkThrow() << "QtAudioDataSourceFactory does not provide an initialisation dialog.";
  return NULL;
}


//-----------------------------------------------------------------------------
IGIConfigurationDialog* QtAudioDataSourceFactory::CreateConfigurationDialog(QWidget *parent,
                                                                                niftk::IGIDataSourceI::Pointer service
                                                                                ) const
{
  mitkThrow() << "QtAudioDataSourceFactory does not provide a configuration dialog.";
  return NULL;
}


//-----------------------------------------------------------------------------
QList<QString> QtAudioDataSourceFactory::GetLegacyClassNames() const
{
  QList<QString> names;
  return names;
}

} // end namespace
