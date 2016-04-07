/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSystemTimeServiceRAII_h
#define niftkSystemTimeServiceRAII_h

#include <niftkIGIServicesExports.h>
#include "niftkSystemTimeServiceI.h"

#include <usServiceReference.h>
#include <usModuleContext.h>

namespace niftk
{

/**
* \class SystemTimeServiceRAII
* \brief RAII object to retrieve the system time via a SystemTimeServiceI implementation.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGISERVICES_EXPORT SystemTimeServiceRAII : public SystemTimeServiceI
{

public:

  /**
  * \brief Obtains service or throws mitk::Exception.
  */
  SystemTimeServiceRAII();

  /**
  * \brief Releases service.
  */
  virtual ~SystemTimeServiceRAII();

  /**
  * \see SystemTimeServiceI::GetSystemTimeInNanoseconds()
  */
  virtual TimeType GetSystemTimeInNanoseconds() const;

private:
  SystemTimeServiceRAII(const SystemTimeServiceRAII&); // deliberately not implemented
  SystemTimeServiceRAII& operator=(const SystemTimeServiceRAII&); // deliberately not implemented

  us::ModuleContext*                                     m_ModuleContext;
  std::vector<us::ServiceReference<SystemTimeServiceI> > m_Refs;
  niftk::SystemTimeServiceI*                             m_Service;
};

} // end namespace

#endif
