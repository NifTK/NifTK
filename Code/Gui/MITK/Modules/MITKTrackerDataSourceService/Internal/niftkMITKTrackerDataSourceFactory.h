/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMITKTrackerDataSourceFactory_h
#define niftkMITKTrackerDataSourceFactory_h

#include <niftkIGIDataSourceFactoryServiceI.h>
#include <mitkSerialCommunication.h>

namespace niftk
{

/**
* \class MITKTrackerDataSourceFactory
* \brief Abstract factory class to create MITKTrackerDataSources.
 */
class MITKTrackerDataSourceFactory : public IGIDataSourceFactoryServiceI
{

public:

  MITKTrackerDataSourceFactory(QString factoryName);
  virtual ~MITKTrackerDataSourceFactory();

  /**
  * \brief Unimplemented pure virtual method, see derived classes.
  * \see IGIDataSourceFactoryServiceI::CreateService()
  */
  virtual IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage,
    const IGIDataSourceProperties& properties) const override = 0;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateInitialisationDialog()
  *
  * All MITK trackers need the port (USB port) number and config file settings at startup.
  */
  virtual IGIInitialisationDialog* CreateInitialisationDialog(QWidget *parent) const override;

  /**
  * \see IGIDataSourceFactoryServiceI::CreateConfigurationDialog()
  *
  * All MITK trackers allow the lag to be set.
  */
  virtual IGIConfigurationDialog* CreateConfigurationDialog(QWidget *parent,
                                                            niftk::IGIDataSourceI::Pointer service
                                                            ) const override;

  /**
  * \brief Returns the empty list, as there are no legacy names.
  */
  virtual QList<QString> GetLegacyClassNames() const override;

  /**
  * \brief Extracts some parameters, needed to construct niftk::NDITrackers.
  */
  void ExtractProperties(const IGIDataSourceProperties& properties,
      mitk::SerialCommunication::PortNumber& outputPortNumber,
      std::string& outputFileName) const;

};

} // end namespace

#endif
