/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceBuffer.h"
#include <mitkExceptionMacro.h>
#include <itkMutexLockHolder.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceBuffer::IGIDataSourceBuffer()
: m_Mutex(itk::FastMutexLock::New())
, m_Lag(0)
, m_FrameRate(0)
, m_Name("")
{
}


//-----------------------------------------------------------------------------
IGIDataSourceBuffer::~IGIDataSourceBuffer()
{
}


//-----------------------------------------------------------------------------
std::string IGIDataSourceBuffer::GetName() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  return m_Name;
}


//-----------------------------------------------------------------------------
void IGIDataSourceBuffer::SetName(const std::string& name)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  m_Name = name;
}


//-----------------------------------------------------------------------------
unsigned int IGIDataSourceBuffer::GetLagInMilliseconds() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  return m_Lag / 1000000; // its stored in nanoseconds.
}


//-----------------------------------------------------------------------------
void IGIDataSourceBuffer::SetLagInMilliseconds(unsigned int milliseconds)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  m_Lag = milliseconds * 1000000; // nanoseconds.
}


//-----------------------------------------------------------------------------
float IGIDataSourceBuffer::GetFrameRate() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  return m_FrameRate;
}


//-----------------------------------------------------------------------------
void IGIDataSourceBuffer::UpdateFrameRate()
{
  unsigned int numberOfFrames = this->GetBufferSize();

  if (numberOfFrames > 1)
  {
    niftk::IGIDataSourceI::IGITimeType  firstTimeStamp = this->GetFirstTimeStamp();
    niftk::IGIDataSourceI::IGITimeType  lastTimeStamp = this->GetLastTimeStamp();
    niftk::IGIDataSourceI::IGITimeType  timeDifference = 0;
    niftk::IGIDataSourceI::IGIIndexType numberOfFrames = 0;

    if (lastTimeStamp < firstTimeStamp)
    {
      mitkThrow() << "Timestamps are not increasing.";
    }

    timeDifference = lastTimeStamp - firstTimeStamp;
    numberOfFrames = this->GetBufferSize();

    itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
    m_FrameRate = (double)1.0 / ((double)timeDifference/(double)(numberOfFrames * 1000000000.0));
  }
  else
  {
    itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
    m_FrameRate = 0;
  }
}

} // end namespace
