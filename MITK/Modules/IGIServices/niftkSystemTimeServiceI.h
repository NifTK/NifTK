/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSystemTimeServiceI_h
#define niftkSystemTimeServiceI_h

#include <niftkIGIServicesExports.h>

#include <mitkServiceInterface.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

/**
* \class SystemTimeServiceI
* \brief Interface for a service that can return the system time.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGISERVICES_EXPORT SystemTimeServiceI
{

public:

  typedef unsigned long long TimeType;
  virtual TimeType GetSystemTimeInNanoseconds() const = 0;

protected:
  SystemTimeServiceI();
  virtual ~SystemTimeServiceI();

private:
  SystemTimeServiceI(const SystemTimeServiceI&); // deliberately not implemented
  SystemTimeServiceI& operator=(const SystemTimeServiceI&); // deliberately not implemented
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::SystemTimeServiceI, "uk.ac.ucl.cmic.SystemTimeServiceI");

#endif
