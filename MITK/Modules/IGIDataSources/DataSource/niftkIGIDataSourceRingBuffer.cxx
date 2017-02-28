/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceRingBuffer.h"
#include <mitkExceptionMacro.h>
#include <itkMutexLockHolder.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceRingBuffer::IGIDataSourceRingBuffer(BufferType::size_type numberOfItems)
: m_FirstItem(-1)
, m_LastItem(-1)
, m_NumberOfItems(numberOfItems)
{
  if (m_NumberOfItems < 1)
  {
    mitkThrow() << "Buffer size should be a number >= 1";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceRingBuffer::~IGIDataSourceRingBuffer()
{
}


//-----------------------------------------------------------------------------
unsigned int IGIDataSourceRingBuffer::GetBufferSize() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  return m_Buffer.size();
}


//-----------------------------------------------------------------------------
void IGIDataSourceRingBuffer::CleanBuffer()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  m_Buffer.clear();
  m_FirstItem = -1;
  m_LastItem = -1;
}


//-----------------------------------------------------------------------------
bool IGIDataSourceRingBuffer::Contains(const niftk::IGIDataSourceI::IGITimeType& time) const
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
int IGIDataSourceRingBuffer::GetNextIndex(const int& currentIndex) const
{
  int result = currentIndex + 1;
  if (result == m_NumberOfItems || result >= m_Buffer.size())
  {
    result = 0;
  }
  return result;
}


//-----------------------------------------------------------------------------
int IGIDataSourceRingBuffer::GetPreviousIndex(const int& currentIndex) const
{
  int result = currentIndex - 1;
  if (result == -1)
  {
    result = m_Buffer.size() -1;
  }
  return result;
}


//-----------------------------------------------------------------------------
void IGIDataSourceRingBuffer::AddToBuffer(std::unique_ptr<niftk::IGIDataType>& item)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size() < m_NumberOfItems)
  {
    m_Buffer.emplace_back(std::move(item));
    m_FirstItem = 0;
    m_LastItem = m_Buffer.size() - 1;
  }
  else
  {
    int nextFrame = this->GetNextIndex(m_LastItem);
    m_Buffer[nextFrame]->Clone(*item);
    m_LastItem = nextFrame;
    m_FirstItem = this->GetNextIndex(m_FirstItem);
  }
}


//-----------------------------------------------------------------------------
niftk::IGIDataSourceI::IGITimeType IGIDataSourceRingBuffer::GetFirstTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size() == 0)
  {
    mitkThrow() << "Empty Buffer, so can't get first time stamp";
  }
  return m_Buffer[m_FirstItem]->GetTimeStampInNanoSeconds();
}


//-----------------------------------------------------------------------------
niftk::IGIDataSourceI::IGITimeType IGIDataSourceRingBuffer::GetLastTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size() == 0)
  {
    mitkThrow() << "Empty Buffer, so can't get last time stamp";
  }
  return m_Buffer[m_LastItem]->GetTimeStampInNanoSeconds();
}


//-----------------------------------------------------------------------------
bool IGIDataSourceRingBuffer::CopyOutItem(const niftk::IGIDataSourceI::IGITimeType& time,
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
  if (m_Buffer[m_FirstItem]->GetTimeStampInNanoSeconds() > effectiveTime)
  {
    return false;
  }

  // If first item in buffer is exactly equal to request time, just return
  // it without searching the buffer. This occurs during playback.
  if (m_Buffer[m_FirstItem]->GetTimeStampInNanoSeconds() == effectiveTime)
  {
    item.Clone(*(m_Buffer[m_FirstItem]));
    return true;
  }

  int bufferIndex = m_FirstItem;
  if (bufferIndex < 0)
  {
    return false;
  }

  while(bufferIndex != m_LastItem && m_Buffer[bufferIndex]->GetTimeStampInNanoSeconds() < effectiveTime)
  {
    bufferIndex = this->GetNextIndex(bufferIndex);
  }

  if (bufferIndex != m_LastItem && m_Buffer[bufferIndex]->GetTimeStampInNanoSeconds() == effectiveTime)
  {
    item.Clone(*(m_Buffer[bufferIndex]));
    return true;
  }

  // Backtrack one step, as we just went past the closest one.
  bufferIndex = this->GetPreviousIndex(bufferIndex);
  item.Clone(*(m_Buffer[bufferIndex]));

  return true;
}

} // end namespace
