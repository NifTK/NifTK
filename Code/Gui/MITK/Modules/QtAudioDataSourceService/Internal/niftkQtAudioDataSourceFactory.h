/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkQtAudioDataSourceFactory_h
#define niftkQtAudioDataSourceFactory_h

#include <niftkIGIDataSourceFactoryServiceI.h>

namespace niftk
{

/**
* \class QtAudioDataSourceFactory
* \brief Factory class to create QtAudioDataSources.
 */
class QtAudioDataSourceFactory : public IGIDataSourceFactoryServiceI
{

public:

  QtAudioDataSourceFactory();
  virtual ~QtAudioDataSourceFactory();

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
  * \throw Always throws mitk::Exception as this source does not have one.
  */
  virtual IGIConfigurationDialog* CreateConfigurationDialog(QWidget *parent,
                                                            niftk::IGIDataSourceI::Pointer service
                                                            ) const override;

  /**
  * \brief Empty list, as we have never really used this one before this architecture upgrade.
  */
  virtual QList<QString> GetLegacyClassNames() const override;
};

} // end namespace

#endif
