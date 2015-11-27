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
*/
class NIFTKIGISERVICES_EXPORT IGIDataSourceFactoryServiceI
{

public:

  virtual IGIDataSourceI::Pointer Create(mitk::DataStorage::Pointer dataStorage) = 0;

  /**
  * \brief Returns the name of the data source, as perceived by the user in the GUI.
  */
  virtual std::string GetDisplayName() const;

  /**
  * \brief Each data source saves data in a folder of a given name.
  */
  virtual std::string GetSaveLocationPrefix() const;

  /**
  * \brief Returns the name of the class that should be instantiated.
  */
  virtual std::string GetNameOfService() const;

  /**
  * \brief Returns the name of the GUI class that should be instantiated.
  */
  virtual std::string GetNameOfGui() const;

  /**
  * \brief Returns true if we need a GUI at startup to configure it.
  */
  virtual bool GetNeedGuiAtStartup() const;

protected:

  IGIDataSourceFactoryServiceI(std::string displayName,
                               std::string savePrefix,
                               std::string service,
                               std::string gui,
                               bool needGuiAtStartup);

  virtual ~IGIDataSourceFactoryServiceI();

private:

  IGIDataSourceFactoryServiceI(const IGIDataSourceFactoryServiceI&); // deliberately not implemented
  IGIDataSourceFactoryServiceI& operator=(const IGIDataSourceFactoryServiceI&); // deliberately not implemented

  std::string m_DisplayName;
  std::string m_SavePrefix;
  std::string m_NameOfService;
  std::string m_NameOfGui;
  bool        m_NeedGuiAtStartup;
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::IGIDataSourceFactoryServiceI, "uk.ac.ucl.cmic.IGIDataSourceFactoryServiceI");

#endif
