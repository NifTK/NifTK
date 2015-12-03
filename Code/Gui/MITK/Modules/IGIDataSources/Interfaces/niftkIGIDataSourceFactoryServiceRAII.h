/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceFactoryServiceRAII_h
#define niftkIGIDataSourceFactoryServiceRAII_h

#include <niftkIGIServicesExports.h>
#include "niftkIGIDataSourceFactoryServiceI.h"
#include "niftkIGIDataSourceI.h"

#include <usServiceReference.h>
#include <usModuleContext.h>

namespace niftk
{

/**
* \class IGIDataSourceFactoryServiceRAII
* \brief RAII object to retrieve a specific IGIDataSourceFactoryServiceI subclass.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGISERVICES_EXPORT IGIDataSourceFactoryServiceRAII
{

public:

  /**
  * \brief Obtains factory service or throws mitk::Exception.
  */
  IGIDataSourceFactoryServiceRAII(const QString& factoryName);

  /**
  * \brief Releases factory service.
  */
  virtual ~IGIDataSourceFactoryServiceRAII();

  /**
  * \brief Can be used to create instances of the requested IGIDataSourceServiceI.
  */
  IGIDataSourceI::Pointer CreateService(mitk::DataStorage::Pointer dataStorage,
                                        const IGIDataSourceProperties& properties);

private:
  IGIDataSourceFactoryServiceRAII(const IGIDataSourceFactoryServiceRAII&); // deliberately not implemented
  IGIDataSourceFactoryServiceRAII& operator=(const IGIDataSourceFactoryServiceRAII&); // deliberately not implemented

  us::ModuleContext*                                               m_ModuleContext;
  std::vector<us::ServiceReference<IGIDataSourceFactoryServiceI> > m_Refs;
  niftk::IGIDataSourceFactoryServiceI*                             m_Service;
};

} // end namespace

#endif
