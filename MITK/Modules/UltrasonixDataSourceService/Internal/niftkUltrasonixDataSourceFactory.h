/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUltrasonixDataSourceFactory_h
#define niftkUltrasonixDataSourceFactory_h

#include <niftkIGIDataSourceFactoryServiceI.h>

namespace niftk
{

/**
* \class UltrasonixDataSourceFactory
* \brief Factory class to create UltrasonixDataSources.
 */
class UltrasonixDataSourceFactory : public IGIDataSourceFactoryServiceI
{

public:

  UltrasonixDataSourceFactory();
  virtual ~UltrasonixDataSourceFactory();

  /**
  * \see IGIDataSourceFactoryServiceI::CreateService()
  */
  virtual IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage,
                                                const IGIDataSourceProperties& properties) const override;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateInitialisationDialog()
  */
  virtual IGIInitialisationDialog* CreateInitialisationDialog(QWidget *parent) const override;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateConfigurationDialog()
  */
  virtual IGIConfigurationDialog* CreateConfigurationDialog(QWidget *parent,
                                                            niftk::IGIDataSourceI::Pointer service
                                                            ) const override;

  /**
  * \brief Returns empty list.
  */
  virtual QList<QString> GetLegacyClassNames() const override;
};

} // end namespace

#endif
