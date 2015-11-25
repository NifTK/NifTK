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
#include "niftkIGIDataSourceServiceI.h"

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

  virtual IGIDataSourceServiceI* Create(mitk::DataStorage::Pointer dataStorage) = 0;

  virtual std::string GetNameOfFactory() const;
  virtual std::string GetNameOfService() const;
  virtual std::string GetNameOfGui() const;

protected:

  IGIDataSourceFactoryServiceI(std::string factory, std::string service, std::string gui);
  virtual ~IGIDataSourceFactoryServiceI();

private:

  IGIDataSourceFactoryServiceI(const IGIDataSourceFactoryServiceI&); // deliberately not implemented
  IGIDataSourceFactoryServiceI& operator=(const IGIDataSourceFactoryServiceI&); // deliberately not implemented

  std::string m_NameOfFactory;
  std::string m_NameOfService;
  std::string m_NameOfGui;
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::IGIDataSourceFactoryServiceI, "uk.ac.ucl.cmic.IGIDataSourceFactoryServiceI");

#endif
