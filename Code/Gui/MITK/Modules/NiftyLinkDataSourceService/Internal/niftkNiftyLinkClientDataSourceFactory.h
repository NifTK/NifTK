/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyLinkClientDataSourceFactory_h
#define niftkNiftyLinkClientDataSourceFactory_h

#include <niftkIGIDataSourceFactoryServiceI.h>

namespace niftk
{

/**
* \class NiftyLinkClientDataSourceFactory
* \brief Factory class to create NiftyLinkClientDataSourceServices.
 */
class NiftyLinkClientDataSourceFactory : public IGIDataSourceFactoryServiceI
{

public:

  NiftyLinkClientDataSourceFactory();
  virtual ~NiftyLinkClientDataSourceFactory();

  /**
  * \see IGIDataSourceFactoryServiceI::CreateService()
  */
  virtual IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const override;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateInitialisationDialog()
  *
  * NiftyLink client services need to connect to a server,
  * and so should present a dialog box to enter IP and port.
  */
  virtual IGIInitialisationDialog* CreateInitialisationDialog(QWidget *parent) const override;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateConfigurationDialog()
  *
  * NiftyLink client services allow the lag to be set.
  */
  virtual IGIConfigurationDialog* CreateConfigurationDialog(QWidget *parent,
                                                            niftk::IGIDataSourceI::Pointer service
                                                            ) const override;

  /**
  * \brief Returns the empty list as there is no legacy equivalent.
  */
  virtual QList<QString> GetLegacyClassNames() const override;

};

} // end namespace

#endif
