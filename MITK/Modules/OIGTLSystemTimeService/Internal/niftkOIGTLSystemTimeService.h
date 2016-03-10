/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkOIGTLSystemTimeService_h
#define niftkOIGTLSystemTimeService_h

#include <niftkSystemTimeServiceI.h>
#include <igtlTimeStamp.h>

namespace niftk
{

/**
* \class OIGTLSystemTimeService
* \brief Implements SystemTimeServiceI using igtl::TimeStamp.
*/
class OIGTLSystemTimeService : public SystemTimeServiceI
{
public:

  OIGTLSystemTimeService();
  ~OIGTLSystemTimeService();

  /**
  * \see SystemTimeServiceI::GetSystemTimeInNanoseconds()
  */
  virtual TimeType GetSystemTimeInNanoseconds() const;

private:

  OIGTLSystemTimeService(const OIGTLSystemTimeService&); // deliberately not implemented
  OIGTLSystemTimeService& operator=(const OIGTLSystemTimeService&); // deliberately not implemented

  igtl::TimeStamp::Pointer m_TimeStamp;
};

} // end namespace

#endif
