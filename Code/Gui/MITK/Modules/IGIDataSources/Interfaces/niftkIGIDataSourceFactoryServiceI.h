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

#include <mitkServiceInterface.h>
#include <mitkDataStorage.h>

namespace niftk
{

/**
* \class IGIDataSourceFactoryServiceI
* \brief Interface for a factory to create niftk::IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*
* Note: Implementors of this interface must be thread-safe.
*
* Note: Deliberately not using Qt datatypes, so that an implementing class does not have to.
*/
class NIFTKIGISERVICES_EXPORT IGIDataSourceFactoryServiceI
{

public:

  virtual IGIDataSourceI::Pointer Create(mitk::DataStorage::Pointer dataStorage) = 0;

  virtual IGIDataSourceI::Pointer Create(const std::string& name,
                                         mitk::DataStorage::Pointer dataStorage) = 0;

  /**
  * \brief Returns the name of the data source, as perceived by the user in the GUI.
  */
  virtual std::string GetName() const;

  /**
  * \brief Returns class names that this source was known as historically.
  */
  virtual std::vector<std::string> GetLegacyClassNames() const = 0;

  /**
  * \brief Returns the name of the service class that should be instantiated.
  */
  virtual std::string GetNameOfService() const;

  /**
  * \brief Returns true if we need a GUI at startup to configure it.
  */
  virtual bool GetNeedGuiAtStartup() const;

  /**
  * \brief Returns the name of the GUI class that should be instantiated at startup.
  */
  virtual std::string GetNameOfStartupGui() const;

  /**
  * \brief Returns the name of the GUI class that should be instantiated
  * to observe the service while it is running.
  */
  virtual std::string GetNameOfObservationGui() const;

protected:

  IGIDataSourceFactoryServiceI(std::string name,
                               std::string service,
                               bool needGuiAtStartup,
                               std::string startupGui,
                               std::string observationGui
                               );

  virtual ~IGIDataSourceFactoryServiceI();

private:

  IGIDataSourceFactoryServiceI(const IGIDataSourceFactoryServiceI&); // deliberately not implemented
  IGIDataSourceFactoryServiceI& operator=(const IGIDataSourceFactoryServiceI&); // deliberately not implemented

  // These should be immutable after construction.
  // Do not provide Setters.
  std::string m_Name; // i.e. name the factory is known as.
  std::string m_NameOfService;
  std::string m_NameOfStartupGui;
  std::string m_NameOfObservationGui;
  bool        m_NeedGuiAtStartup;
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::IGIDataSourceFactoryServiceI, "uk.ac.ucl.cmic.IGIDataSourceFactoryServiceI");

#endif
