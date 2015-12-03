/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceFactoryServiceI_h
#define niftkIGIDataSourceFactoryServiceI_h

#include <niftkIGIServicesExports.h>
#include <niftkIGIDataSourceI.h>
#include <niftkIGIConfigurationDialog.h>
#include <niftkIGIInitialisationDialog.h>

#include <mitkServiceInterface.h>
#include <mitkDataStorage.h>

#include <QWidget>
#include <QMap>
#include <QString>

namespace niftk
{

/**
* \class IGIDataSourceFactoryServiceI
* \brief Interface for a factory to create niftk::IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGISERVICES_EXPORT IGIDataSourceFactoryServiceI
{

public:

  /**
  * \brief Creates the actual data source service.
  */
  virtual IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage, const QMap<QString, QVariant>& properties) const = 0;

  /**
  * \brief Creates the dialog box used to initialise the service.
  */
  virtual IGIInitialisationDialog* CreateInitialisationDialog(QWidget *parent) const = 0;

  /**
  * \brief Creates the dialog box used to configure the service while its running.
  */
  virtual IGIConfigurationDialog* CreateConfigurationDialog(QWidget *parent, niftk::IGIDataSourceI::Pointer) const = 0;

  /**
  * \brief Returns the name of the data source factory, as perceived by the user in the GUI.
  */
  virtual std::string GetName() const;

  /**
  * \brief Returns class names that this source was known as historically.
  */
  virtual std::vector<std::string> GetLegacyClassNames() const = 0;

  bool HasInitialiseGui() const;
  bool HasConfigurationGui() const;

protected:

  IGIDataSourceFactoryServiceI(std::string name,
                               bool hasInitialiseGui,
                               bool hasConfigurationGui);
  virtual ~IGIDataSourceFactoryServiceI();

private:

  IGIDataSourceFactoryServiceI(const IGIDataSourceFactoryServiceI&); // deliberately not implemented
  IGIDataSourceFactoryServiceI& operator=(const IGIDataSourceFactoryServiceI&); // deliberately not implemented

  std::string m_Name;
  bool        m_HasInitialiseGui;
  bool        m_HasConfigurationGui;
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::IGIDataSourceFactoryServiceI, "uk.ac.ucl.cmic.IGIDataSourceFactoryServiceI");

#endif
