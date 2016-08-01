/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOIGTLSystemTimeService.h"

namespace niftk
{

//-----------------------------------------------------------------------------
OIGTLSystemTimeService::OIGTLSystemTimeService()
: m_Mutex(itk::FastMutexLock::New())
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
  m_TimeStamp = igtl::TimeStamp::New();
  m_TimeStamp->GetTime();
}


//-----------------------------------------------------------------------------
OIGTLSystemTimeService::~OIGTLSystemTimeService()
{

}


//-----------------------------------------------------------------------------
SystemTimeServiceI::TimeType OIGTLSystemTimeService::GetSystemTimeInNanoseconds() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
  m_TimeStamp->GetTime();
  return m_TimeStamp->GetTimeStampInNanoseconds();
}

} // end namespace
