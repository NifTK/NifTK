/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceLinearBuffer.h"
#include <itkMutexLockHolder.h>
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceLinearBuffer::IGIDataSourceLinearBuffer(BufferType::size_type minSize)
: m_MinimumSize(minSize)
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
IGIDataSourceLinearBuffer::~IGIDataSourceLinearBuffer()
{
}


//-----------------------------------------------------------------------------
unsigned int IGIDataSourceLinearBuffer::GetBufferSize() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  return m_Buffer.size();
}


//-----------------------------------------------------------------------------
void IGIDataSourceLinearBuffer::DestroyBuffer()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  m_Buffer.clear();
}


//-----------------------------------------------------------------------------
void IGIDataSourceLinearBuffer::CleanBuffer()
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
      ++endIter;
      ++counter;
    }

    if (counter > 1 && startIter != endIter)
    {
      m_Buffer.erase(startIter, endIter);
    }
  }
}


//-----------------------------------------------------------------------------
bool IGIDataSourceLinearBuffer::Contains(const niftk::IGIDataSourceI::IGITimeType& time) const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size() == 0)
  {
    return false;
  }

  bool containsIt = false;

  BufferType::const_iterator iter = m_Buffer.begin();
  while(iter != m_Buffer.end())
  {
    if ((*iter)->GetTimeStampInNanoSeconds() == time)
    {
      containsIt = true;
      break;
    }
    ++iter;
  }
  return containsIt;
}


//-----------------------------------------------------------------------------
void IGIDataSourceLinearBuffer::AddToBuffer(std::unique_ptr<niftk::IGIDataType>& item)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  m_Buffer.push_back(std::move(item));
}


//-----------------------------------------------------------------------------
niftk::IGIDataSourceI::IGITimeType IGIDataSourceLinearBuffer::GetFirstTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size() == 0)
  {
    mitkThrow() << "Empty Buffer, so can't get first time stamp";
  }

  return (*m_Buffer.begin())->GetTimeStampInNanoSeconds();
}


//-----------------------------------------------------------------------------
niftk::IGIDataSourceI::IGITimeType IGIDataSourceLinearBuffer::GetLastTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size() == 0)
  {
    mitkThrow() << "Empty Buffer, so can't get last time stamp";
  }

  return (*(--(m_Buffer.end())))->GetTimeStampInNanoSeconds();
}


//-----------------------------------------------------------------------------
bool IGIDataSourceLinearBuffer::CopyOutItem(const niftk::IGIDataSourceI::IGITimeType& time,
                                            niftk::IGIDataType& item) const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (time < m_Lag)
  {
    mitkThrow() << "The requested time " << time
                << " is obviously too small, suggesting a programming bug." << std::endl;
  }

  if (m_Buffer.size() == 0)
  {
    return false;
  }

  niftk::IGIDataSourceI::IGITimeType effectiveTime = time - m_Lag; // normally lag is zero.

  // If first item in buffer is later than requested time,
  // we don't have any data early enough, so abandon.
  if ((*(m_Buffer.begin()))->GetTimeStampInNanoSeconds() > effectiveTime)
  {
    return false;
  }

  // If first item in buffer is exactly equal to request time, just return
  // it without searching the buffer. This occurs during playback.
  if ((*(m_Buffer.begin()))->GetTimeStampInNanoSeconds() == effectiveTime)
  {
    item.Clone(*(*(m_Buffer.begin())));
    return true;
  }

  BufferType::const_iterator iter = m_Buffer.begin();
  while(iter != m_Buffer.end() && (*iter)->GetTimeStampInNanoSeconds() < effectiveTime)
  {
    ++iter;
  }

  if (iter != m_Buffer.end() && (*iter)->GetTimeStampInNanoSeconds() == effectiveTime)
  {
    item.Clone(*(*iter));
    return true;
  }

  --iter; // Backtrack one step, as we just went past the closest one.
  item.Clone(*(*iter));
  return true;
}

} // end namespace
