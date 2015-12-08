/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyLinkServerDataSourceFactory_h
#define niftkNiftyLinkServerDataSourceFactory_h

#include <niftkIGIDataSourceFactoryServiceI.h>

namespace niftk
{

/**
* \class NiftyLinkServerDataSourceFactory
* \brief Factory class to create NiftyLinkServerDataSourceServices.
 */
class NiftyLinkServerDataSourceFactory : public IGIDataSourceFactoryServiceI
{

public:

  NiftyLinkServerDataSourceFactory();
  virtual ~NiftyLinkServerDataSourceFactory();

  /**
  * \see IGIDataSourceFactoryServiceI::CreateService()
  */
  virtual IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const override;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateInitialisationDialog()
  *
  * NiftyLink server services need to specify a port to listen on.
  */
  virtual IGIInitialisationDialog* CreateInitialisationDialog(QWidget *parent) const override;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateConfigurationDialog()
  *
  * NiftyLink server services allow the lag to be set.
  */
  virtual IGIConfigurationDialog* CreateConfigurationDialog(QWidget *parent,
                                                            niftk::IGIDataSourceI::Pointer service
                                                            ) const override;

  /**
  * \brief Returns the empty list as there is no legacy equivalent.
  *
  * To replay the data from the legacy "network tracker", or "network ultrasonix"
  * data sources, you should use the MITK local trackers that can read .txt files.
  */
  virtual QList<QString> GetLegacyClassNames() const override;

};

} // end namespace

#endif
