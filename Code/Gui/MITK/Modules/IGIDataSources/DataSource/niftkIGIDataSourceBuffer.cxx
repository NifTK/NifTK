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
, m_MinimumSize(minSize)
, m_FrameRate(0)
, m_Lag(0)
{
  if (minSize == 0)
  {
    mitkThrow() << "Buffer size should be a number > 0";
  }
  m_BufferIterator = m_Buffer.begin();
}


//-----------------------------------------------------------------------------
IGIDataSourceBuffer::~IGIDataSourceBuffer()
{
}


//-----------------------------------------------------------------------------
void IGIDataSourceBuffer::SetLagInMilliseconds(unsigned int milliseconds)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  m_Lag = milliseconds * 1000000; // nanoseconds.
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceBuffer::AddToBuffer(niftk::IGIDataType::Pointer item)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  m_Buffer.insert(item);
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
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size() > m_MinimumSize)
  {
    BufferType::size_type numberToDelete =  m_Buffer.size() - m_MinimumSize;
    BufferType::size_type counter = 0;

    BufferType::iterator startIter = m_Buffer.begin();
    BufferType::iterator endIter = m_Buffer.begin();

    while(endIter != m_Buffer.end() && counter < numberToDelete)
    {
      endIter++;
      counter++;
    }

    if (numberToDelete > 1 && startIter != endIter)
    {
      m_Buffer.erase(startIter, endIter);
      this->Modified();
    }
  }
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
  if (m_Buffer.size() > 1)
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


//-----------------------------------------------------------------------------
niftk::IGIDataType::Pointer IGIDataSourceBuffer::GetItem(const niftk::IGIDataType::IGITimeType& time) const
{
  if (time < m_Lag)
  {
    mitkThrow() << "The requested time " << time << " is obviously too small, suggesting a programming bug." << std::endl;
  }

  niftk::IGIDataType::Pointer result = NULL;

  if (m_Buffer.size() < 2)
  {
    return result;
  }

  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  niftk::IGIDataType::IGITimeType effectiveTime = time - m_Lag; // normally lag is zero.

  // If first item in buffer is later than requested time,
  // we don't have any data early enough, so abandon.
  if ((*(m_Buffer.begin()))->GetTimeStampInNanoSeconds() > effectiveTime)
  {
    return result;
  }

  BufferType::iterator iter = m_Buffer.begin();
  while(iter != m_Buffer.end() && (*iter)->GetTimeStampInNanoSeconds() <= effectiveTime)
  {
    iter++;
  }

  // If we stopped because we hit the end of the buffer, then
  // there is no data close enough. We could just return the last one,
  // but Id rather be overcautious, and return nothing. The side
  // effect of this, is that if the last item, was exactly equal
  // to the requested time stamp, we wont retrieve it.
  if (iter == m_Buffer.end())
  {
    return result;
  }

  // Backtrack one step, as we just went past the closest one.
  iter--;
  return *iter;
}

} // end namespace

