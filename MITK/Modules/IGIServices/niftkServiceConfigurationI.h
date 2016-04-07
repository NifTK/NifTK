/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkServiceConfigurationI_h
#define niftkServiceConfigurationI_h

#include <niftkIGIServicesExports.h>
#include <usServiceProperties.h>

namespace niftk
{

/**
* \class ServiceConfigurationI
* \brief Interface to describe how any service should be configured.
*/
class NIFTKIGISERVICES_EXPORT ServiceConfigurationI
{

public:

  /**
  * \brief Enables clients to pass arbitrary parameters to configure the service.
  * \throws mitk::Exception for all errors.
  */
  virtual void Configure(const us::ServiceProperties& properties) = 0;

protected:
  ServiceConfigurationI() {}
  virtual ~ServiceConfigurationI() {}

private:
  ServiceConfigurationI(const ServiceConfigurationI&); // deliberately not implemented
  ServiceConfigurationI& operator=(const ServiceConfigurationI&); // deliberately not implemented
};

} // end namespace

#endif
