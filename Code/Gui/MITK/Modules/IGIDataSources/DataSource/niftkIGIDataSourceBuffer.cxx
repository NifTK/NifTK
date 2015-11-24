/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceBuffer.h"
#include <itkMutexLockHolder.h>
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
bool IGIDataSourceBuffer::TimeStampComparator::operator()(const niftk::IGIDataType::Pointer& a, const niftk::IGIDataType::Pointer& b)
{
  assert(a.IsNotNull());
  assert(b.IsNotNull());
  return a->GetTimeStampInNanoSeconds() < b->GetTimeStampInNanoSeconds();
}


//-----------------------------------------------------------------------------
IGIDataSourceBuffer::IGIDataSourceBuffer(BufferType::size_type minSize)
: m_Mutex(itk::FastMutexLock::New())
, m_TimeCreated(NULL)
, m_MinimumSize(minSize)
, m_FrameRate(0)
{
  m_TimeCreated = igtl::TimeStamp::New();
  m_TimeCreated->GetTime();
}


//-----------------------------------------------------------------------------
IGIDataSourceBuffer::~IGIDataSourceBuffer()
{
}


//-----------------------------------------------------------------------------
void IGIDataSourceBuffer::AddToBuffer(niftk::IGIDataType::Pointer item)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  this->UpdateFrameRate();
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceBuffer::ClearBuffer()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
  m_Buffer.clear();
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceBuffer::CleanBuffer()
{
  this->Modified();
}


//-----------------------------------------------------------------------------
IGIDataSourceBuffer::BufferType::size_type IGIDataSourceBuffer::GetBufferSize() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
  return m_Buffer.size();
}


//-----------------------------------------------------------------------------
niftk::IGIDataType::IGITimeType IGIDataSourceBuffer::GetFirstTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size())
  {
    mitkThrow() << "Empty Buffer, so can't get first time stamp";
  }

  return (*m_Buffer.begin())->GetTimeStampInNanoSeconds();
}


//-----------------------------------------------------------------------------
niftk::IGIDataType::IGITimeType IGIDataSourceBuffer::GetLastTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size())
  {
    mitkThrow() << "Empty Buffer, so can't get first last stamp";
  }

  return (*(--(m_Buffer.end())))->GetTimeStampInNanoSeconds();
}


//-----------------------------------------------------------------------------
void IGIDataSourceBuffer::UpdateFrameRate()
{
  if (m_Buffer.size() > 2)
  {
    niftk::IGIDataType::IGITimeType  firstTimeStamp = 0;
    niftk::IGIDataType::IGIIndexType firstFrameId = 0;
    niftk::IGIDataType::IGITimeType  lastTimeStamp = 0;
    niftk::IGIDataType::IGIIndexType lastFrameId = 0;
    niftk::IGIDataType::IGITimeType  timeDifference = 0;
    niftk::IGIDataType::IGIIndexType numberOfFrames = 0;

    firstTimeStamp = (*m_Buffer.begin())->GetTimeStampInNanoSeconds();
    firstFrameId = (*m_Buffer.begin())->GetFrameId(); // assumed to be sequentially increasing

    BufferType::iterator iter = m_Buffer.end();
    iter--;

    lastTimeStamp = (*iter)->GetTimeStampInNanoSeconds();
    lastFrameId = (*iter)->GetFrameId(); // assumed to be sequentially increasing

    if (lastTimeStamp < firstTimeStamp)
    {
      mitkThrow() << "Timestamps are not increasing.";
    }
    if (lastFrameId < firstFrameId)
    {
      mitkThrow() << "FrameIds are not increasing.";
    }

    timeDifference = lastTimeStamp - firstTimeStamp;
    numberOfFrames = lastFrameId - firstFrameId;

    m_FrameRate = (double)1.0 / ((double)timeDifference/(double)(numberOfFrames * 1000000000.0));
  }
}


//-----------------------------------------------------------------------------
float IGIDataSourceBuffer::GetFrameRate() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
  return m_FrameRate;
}

} // end namespace

