/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNVidiaSDIDataSourceFactory_h
#define niftkNVidiaSDIDataSourceFactory_h

#include <niftkIGIDataSourceFactoryServiceI.h>

namespace niftk
{

/**
* \class NVidiaSDIDataSourceFactory
* \brief Factory class to create NVidiaSDIDataSources.
 */
class NVidiaSDIDataSourceFactory : public IGIDataSourceFactoryServiceI
{

public:

  NVidiaSDIDataSourceFactory();
  virtual ~NVidiaSDIDataSourceFactory();

  /**
  * \see IGIDataSourceFactoryServiceI::CreateService()
  */
  virtual IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage,
                                                const IGIDataSourceProperties& properties) const override;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateInitialisationDialog()
  * \throw Always throws mitk::Exception as this source does not have one.
  */
  virtual IGIInitialisationDialog* CreateInitialisationDialog(QWidget *parent) const override;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateConfigurationDialog()
  */
  virtual IGIConfigurationDialog* CreateConfigurationDialog(QWidget *parent,
                                                            niftk::IGIDataSourceI::Pointer service
                                                            ) const override;

  /**
  * \brief Returns "QmitkIGIOpenCVDataSource".
  */
  virtual QList<QString> GetLegacyClassNames() const override;
};

} // end namespace

#endif
